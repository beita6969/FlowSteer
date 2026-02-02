# -*- coding: utf-8 -*-
# @Desc: Formatter system for structured LLM responses
# Adapted from AFlow's formatter.py

from typing import Dict, List, Tuple, Type, Optional, Union, Any
from pydantic import BaseModel, Field, create_model
import re
from abc import ABC, abstractmethod


class FormatError(Exception):
    """Exception raised when response format validation fails"""
    pass


class BaseFormatter(BaseModel):
    """Base class for all formatters"""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def prepare_prompt(self, prompt: str) -> str:
        """Prepare the prompt to instruct the LLM to return in the required format"""
        pass

    @abstractmethod
    def validate_response(self, response: str) -> Tuple[bool, Any]:
        """Validate if the response matches the expected format"""
        pass

    def format_error_message(self) -> str:
        """Return an error message for invalid format"""
        return f"Response did not match the expected {self.__class__.__name__} format"


class XmlFormatter(BaseFormatter):
    """Formatter for XML responses"""
    model: Optional[Type[BaseModel]] = None
    fields: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, fields_dict: Dict[str, str]) -> "XmlFormatter":
        """
        Create formatter from a dictionary of field names and descriptions

        Args:
            fields_dict: Dictionary where keys are field names and values are field descriptions

        Returns:
            An XmlFormatter instance configured with the specified fields
        """
        model_fields = {}
        for name, desc in fields_dict.items():
            model_fields[name] = (str, Field(default="", description=desc))

        model_class = create_model("XmlResponseModel", **model_fields)

        return cls(model=model_class)

    @classmethod
    def from_model(cls, model_class: Type[BaseModel]) -> "XmlFormatter":
        """
        Create formatter from an existing Pydantic model class

        Args:
            model_class: A Pydantic model class

        Returns:
            An XmlFormatter instance configured with the model's fields
        """
        return cls(model=model_class)

    def _get_field_names(self) -> List[str]:
        """Get field names from the model"""
        if self.model:
            return list(self.model.model_fields.keys())
        return []

    def _get_field_description(self, field_name: str) -> str:
        """Get field description from the model"""
        if self.model and field_name in self.model.model_fields:
            return self.model.model_fields[field_name].description or ""
        return ""

    def prepare_prompt(self, prompt: str) -> str:
        """Prepare prompt with XML format instructions"""
        examples = []
        for field_name in self._get_field_names():
            description = self._get_field_description(field_name)
            examples.append(f"<{field_name}>{description}</{field_name}>")

        example_str = "\n".join(examples)

        instructions = prompt + f"\n# Response format (must be strictly followed) (do not include any other formats except for the given XML format):\n{example_str}"
        return instructions

    def validate_response(self, response: str) -> Tuple[bool, dict]:
        """Validate if the response contains all required fields in XML format"""
        try:
            # Be defensive: some OpenAI-compatible servers may return non-str content.
            if response is None:
                response = ""
            elif isinstance(response, bytes):
                response = response.decode("utf-8", "ignore")
            elif not isinstance(response, str):
                response = str(response)

            # Pattern to match XML tags: <tag>content</tag>
            pattern = r"<(\w+)>(.*?)</\1>"
            matches = re.findall(pattern, response, re.DOTALL)

            found_fields = {match[0]: match[1].strip() for match in matches}

            if not found_fields:
                # Special-case: voting operators sometimes reply with a bare letter like "A".
                field_names = set(self._get_field_names())
                if "solution_letter" in field_names:
                    raw = response.strip()
                    if raw:
                        # 1) If the whole response is (almost) a single letter.
                        m = re.fullmatch(r"([A-Za-z])[\s\.\)\]]*", raw)
                        if m:
                            return True, {"solution_letter": m.group(1).upper()}

                        # 2) Try to extract the last standalone letter token (often the chosen option).
                        tokens = re.findall(r"\b([A-Za-z])\b", raw)
                        if tokens:
                            return True, {"solution_letter": tokens[-1].upper()}

                raise FormatError("No valid XML tags found in response. Response may not be in expected format.")

            # Check required fields
            for field_name in self._get_field_names():
                if self.model:
                    field = self.model.model_fields[field_name]
                    # Check if field is required (no default value)
                    is_required = field.default is None and field.default_factory is None

                    if is_required and (field_name not in found_fields or not found_fields[field_name]):
                        raise FormatError(f"Field '{field_name}' is missing or empty.")

            return True, found_fields
        except FormatError:
            raise
        except Exception as e:
            return False, None

    def format_error_message(self) -> str:
        """Return a helpful error message for XML format"""
        field_names = self._get_field_names()
        return f"Response must contain XML tags for fields: {', '.join(field_names)}"


class CodeFormatter(BaseFormatter):
    """
    Formatter for extracting and sanitizing code from LLM responses.
    Handles both markdown code blocks and raw code responses.
    """

    function_name: Optional[str] = None

    def prepare_prompt(self, prompt: str) -> str:
        """
        Prepare the prompt to instruct the LLM to return code in a proper format.

        Args:
            prompt: The original prompt

        Returns:
            The prompt with instructions to return code in markdown format
        """
        code_instructions = (
            "\n\n"
            "Please write your code solution in Python. "
            "Return ONLY the complete, runnable code without explanations. "
            "Use proper Python syntax and formatting. "
        )

        if self.function_name:
            code_instructions += (
                f"\nMake sure to include a function named '{self.function_name}' in your solution. "
                f"This function will be the entry point for the program."
            )

        return prompt + code_instructions

    def validate_response(self, response: str) -> Tuple[bool, Union[Dict[str, str], str, None]]:
        """
        Extract code from response and validate it.

        Args:
            response: The LLM response

        Returns:
            A tuple with (is_valid, extracted_code_dict)
        """
        try:
            # First try to extract code from markdown code blocks
            code = self._extract_code_from_markdown(response)

            # If no code blocks found, try to extract code from mixed content
            if not code:

                code = self._extract_code_from_mixed_content(response)

            # If still no code, check if response looks like Python code
            if not code:
                if self._looks_like_python_code(response):
                    code = response
                else:
                    # Response is explanation text, not code - return error
                    return False, {"error": "Response contains explanation text instead of Python code"}

            # Basic validation - check if code is not empty
            if not code.strip():
                return False, None

            # Optionally sanitize the code (remove dangerous imports)
            sanitized_code = self._sanitize_code(code)

            # Return the sanitized code
            result = {"response": sanitized_code, "code": sanitized_code}
            return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def _looks_like_python_code(self, text: str) -> bool:
        """
        Check if text looks like Python code (not explanation text).

        Returns True if text contains Python code indicators.
        """
        # Python code indicators
        code_patterns = [
            r'^\s*def\s+\w+\s*\(',      # function definition
            r'^\s*class\s+\w+',          # class definition
            r'^\s*import\s+\w+',         # import statement
            r'^\s*from\s+\w+\s+import',  # from import
            r'^\s*if\s+__name__',        # main guard
            r'^\s*return\s+',            # return statement at start of line
            r'^\s*for\s+\w+\s+in\s+',    # for loop
            r'^\s*while\s+',             # while loop
            r'^\s*try\s*:',              # try block
            r'^\s*@\w+',                 # decorator
        ]

        # Explanation text indicators (should NOT be treated as code)
        explanation_patterns = [
            r'^Step\s*\d+\s*:',          # "Step 1:", "Step 2:"
            r'^First,\s+',               # "First, we..."
            r'^To\s+solve\s+',           # "To solve this..."
            r'^The\s+solution\s+',       # "The solution is..."
            r'^Here\s+is\s+',            # "Here is the..."
            r'^Let\s+me\s+',             # "Let me explain..."
            r'^\d+\.\s+\w+',             # "1. First step"
        ]

        # Check for explanation patterns first (higher priority)
        for pattern in explanation_patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return False

        # Check for code patterns
        for pattern in code_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True

        return False

    def _extract_code_from_markdown(self, text: str) -> str:
        """
        Extract code from markdown code blocks in the response.

        Args:
            text: The text containing possible markdown code blocks

        Returns:
            The extracted code as a string, or empty string if no code blocks found
        """
        # Look for Python code blocks (```python ... ```)
        python_pattern = r"```python\s*([\s\S]*?)\s*```"
        python_matches = re.findall(python_pattern, text)

        if python_matches:
            return "\n\n".join(python_matches)

        # If no Python blocks found, look for generic code blocks (``` ... ```)
        generic_pattern = r"```\s*([\s\S]*?)\s*```"
        generic_matches = re.findall(generic_pattern, text)

        if generic_matches:
            return "\n\n".join(generic_matches)

        return ""

    def _extract_code_from_mixed_content(self, text: str) -> str:
        """

         LLM 

        Args:
            text: 

        Returns:
        """
        if not text:
            return ""

        lines = text.split('\n')
        code_lines = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            if not in_code_block:
                code_start_patterns = [
                    r'^(import|from)\s+\w+',
                    r'^def\s+\w+\s*\(',
                    r'^class\s+\w+',
                    r'^@\w+',
                    r'^#\s*-\*-',
                ]
                for pattern in code_start_patterns:
                    if re.match(pattern, stripped):
                        in_code_block = True
                        break

            if in_code_block:
                end_patterns = [
                    r'^(Step\s*\d+|First,|To\s+solve|The\s+solution|Here\s+is|Let\s+me|Note:|Output:|Result:|Answer:)',
                ]
                is_end = False
                for pattern in end_patterns:
                    if re.match(pattern, stripped, re.IGNORECASE):
                        is_end = True
                        break

                if is_end and code_lines:
                    break

                code_lines.append(line)

        if code_lines:
            code = '\n'.join(code_lines).strip()
            if 'def ' in code or 'import ' in code:
                return code

        return ""

    def _sanitize_code(self, code: str) -> str:
        """Basic code sanitization - can be enhanced with full sanitize.py"""
        # List of disallowed imports
        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh",
            "ggplot", "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                # Remove the import line instead of failing
                code = re.sub(rf'^.*(?:import {lib}|from {lib}).*$', '', code, flags=re.MULTILINE)

        return code.strip()

    def format_error_message(self) -> str:
        """Return a helpful error message if code validation fails"""
        base_message = "Could not extract valid Python code from the response."
        if self.function_name:
            return f"{base_message} Make sure the code includes a function named '{self.function_name}'."
        return base_message

    @classmethod
    def create(cls, function_name: Optional[str] = None) -> "CodeFormatter":
        """
        Factory method to create a CodeFormatter instance

        Args:
            function_name: Optional name of the function to extract

        Returns:
            A configured CodeFormatter instance
        """
        return cls(function_name=function_name)


class TextFormatter(BaseFormatter):
    """Simple formatter for plain text responses - no validation needed"""

    def prepare_prompt(self, prompt: str) -> str:
        """Return prompt unchanged for plain text"""
        return prompt

    def validate_response(self, response: str) -> Tuple[bool, Union[str, None]]:
        """
        For plain text formatter, we simply return the response as is without validation
        since there are no format restrictions
        """
        return True, response


# Export all classes
__all__ = [
    'FormatError',
    'BaseFormatter',
    'XmlFormatter',
    'CodeFormatter',
    'TextFormatter'
]
