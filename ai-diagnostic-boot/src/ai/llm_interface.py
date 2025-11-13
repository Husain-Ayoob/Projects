"""
LLM interface for AI-powered diagnostics.
Handles communication with llama.cpp for issue analysis and fix generation.
"""

import subprocess
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class LLMInterface:
    """Interface for LLM-based diagnostic analysis."""

    def __init__(self, model_path: str = "/system/model.gguf",
                 system_prompt_path: str = "/system/system-prompt.txt",
                 context_window: int = 8192,
                 temperature: float = 0.3,
                 max_tokens: int = 2048,
                 n_threads: int = 4):
        """
        Initialize LLM interface.

        Args:
            model_path: Path to GGUF model file
            system_prompt_path: Path to system prompt file
            context_window: Context window size in tokens
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            n_threads: Number of CPU threads to use
        """
        self.model_path = model_path
        self.system_prompt_path = system_prompt_path
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_threads = n_threads
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            with open(self.system_prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            # Default system prompt if file not found
            return """You are an expert IT diagnostic AI assistant. Your role is to:
1. Analyze system diagnostic data
2. Identify hardware and software issues
3. Rank issues by severity (critical, high, medium, low)
4. Generate fix scripts (bash or python) for each issue
5. Provide clear recommendations

Output format must be valid JSON with this structure:
{
  "analysis": "Brief analysis summary",
  "issues": [
    {
      "issue": "Description",
      "severity": "critical|high|medium|low",
      "diagnosis": "Detailed diagnosis",
      "fix_script": "#!/bin/bash\\ncommand here",
      "risk_level": "low|medium|high",
      "estimated_time": "time estimate"
    }
  ]
}

Be conservative with fixes. Only suggest safe operations. Always include error handling."""

    def analyze_diagnostics(self, diagnostic_data: str) -> Tuple[Dict, float]:
        """
        Analyze diagnostic data using LLM.

        Args:
            diagnostic_data: Formatted diagnostic report string

        Returns:
            Tuple of (analysis_result, inference_time)
        """
        start_time = time.time()

        # Build prompt
        prompt = f"{self.system_prompt}\n\n"
        prompt += "Analyze the following system diagnostic report and provide fix recommendations:\n\n"
        prompt += diagnostic_data
        prompt += "\n\nProvide your analysis in JSON format as specified."

        # Call LLM (using llama-cpp-python or direct llama.cpp)
        response = self._call_llm(prompt)

        inference_time = time.time() - start_time

        # Parse response
        try:
            analysis = self._parse_response(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                'analysis': 'Failed to parse LLM response',
                'issues': [],
                'raw_response': response
            }

        return analysis, inference_time

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM using llama.cpp.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        # Check if model exists
        if not Path(self.model_path).exists():
            return json.dumps({
                'analysis': 'LLM model not found',
                'issues': [],
                'error': f'Model file not found at {self.model_path}'
            })

        try:
            # Try using llama-cpp-python if available
            try:
                from llama_cpp import Llama

                llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_window,
                    n_threads=self.n_threads,
                    n_gpu_layers=0  # CPU only for MVP
                )

                output = llm(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=["</s>", "###"],
                    echo=False
                )

                return output['choices'][0]['text']

            except ImportError:
                # Fallback to direct llama.cpp binary call
                return self._call_llama_cpp_binary(prompt)

        except Exception as e:
            # Return error as JSON
            return json.dumps({
                'analysis': 'LLM inference failed',
                'issues': [],
                'error': str(e)
            })

    def _call_llama_cpp_binary(self, prompt: str) -> str:
        """
        Call llama.cpp binary directly.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        # Write prompt to temp file
        prompt_file = "/tmp/llm_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        try:
            # Call llama.cpp main binary
            result = subprocess.run(
                [
                    'llama.cpp',
                    '-m', self.model_path,
                    '-f', prompt_file,
                    '-n', str(self.max_tokens),
                    '-t', str(self.n_threads),
                    '--temp', str(self.temperature),
                    '-c', str(self.context_window)
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            return result.stdout

        except subprocess.TimeoutExpired:
            return json.dumps({
                'analysis': 'LLM inference timed out',
                'issues': [],
                'error': 'Inference took too long (>5 minutes)'
            })
        except Exception as e:
            return json.dumps({
                'analysis': 'LLM call failed',
                'issues': [],
                'error': str(e)
            })

    def _parse_response(self, response: str) -> Dict:
        """
        Parse LLM response into structured format.

        Args:
            response: Raw LLM response

        Returns:
            Parsed analysis dictionary
        """
        # Try to find JSON in response
        # LLMs sometimes include extra text before/after JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)

        # If no JSON found, raise error
        raise json.JSONDecodeError("No JSON found in response", response, 0)

    def generate_fix_for_issue(self, issue_description: str,
                               system_context: str) -> Tuple[Dict, float]:
        """
        Generate a specific fix for a single issue.

        Args:
            issue_description: Description of the issue
            system_context: Relevant system context

        Returns:
            Tuple of (fix_data, inference_time)
        """
        start_time = time.time()

        prompt = f"{self.system_prompt}\n\n"
        prompt += f"Generate a fix script for the following issue:\n\n"
        prompt += f"Issue: {issue_description}\n\n"
        prompt += f"System Context:\n{system_context}\n\n"
        prompt += "Provide a safe fix script in JSON format:\n"
        prompt += '{"fix_script": "#!/bin/bash\\n...", "explanation": "...", "risk_level": "low|medium|high"}'

        response = self._call_llm(prompt)
        inference_time = time.time() - start_time

        try:
            fix_data = self._parse_response(response)
        except json.JSONDecodeError:
            fix_data = {
                'fix_script': '',
                'explanation': 'Failed to generate fix',
                'risk_level': 'high',
                'error': 'Could not parse LLM response'
            }

        return fix_data, inference_time

    def validate_fix_script(self, script: str, issue: str) -> Tuple[bool, str]:
        """
        Ask LLM to validate a fix script.

        Args:
            script: Fix script to validate
            issue: Issue the script addresses

        Returns:
            Tuple of (is_safe, explanation)
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "Validate the following fix script for safety:\n\n"
        prompt += f"Issue: {issue}\n\n"
        prompt += f"Script:\n{script}\n\n"
        prompt += "Respond with JSON: "
        prompt += '{"safe": true/false, "explanation": "...", "concerns": [...]}'

        response = self._call_llm(prompt)

        try:
            validation = self._parse_response(response)
            is_safe = validation.get('safe', False)
            explanation = validation.get('explanation', 'No explanation provided')
            return is_safe, explanation
        except json.JSONDecodeError:
            return False, "Could not validate script - LLM response invalid"

    def get_model_info(self) -> Dict:
        """Get information about loaded model."""
        return {
            'model_path': self.model_path,
            'exists': Path(self.model_path).exists(),
            'context_window': self.context_window,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'n_threads': self.n_threads
        }
