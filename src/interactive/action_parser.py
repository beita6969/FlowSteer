import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    ADD = "add"
    DELETE = "delete"
    MODIFY = "modify"
    SET_PROMPT = "set_prompt"
    FINISH = "finish"
    INVALID = "invalid"


class StructureType(Enum):
    OPERATOR = "operator"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


@dataclass
class ParsedAction:
    """"""
    action_type: ActionType
    structure_type: Optional[StructureType] = None
    operator: Optional[str] = None
    operators: Optional[List[str]] = None
    target: Optional[str] = None
    condition: Optional[str] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None
    loop_count: int = 3
    position: Optional[int] = None
    custom_prompt: Optional[str] = None
    raw_text: str = ""
    parse_error: Optional[str] = None
    reasoning: Optional[str] = None
    final_answer: Optional[str] = None

    def is_valid(self) -> bool:
        """"""
        return self.action_type != ActionType.INVALID and self.parse_error is None

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            "action_type": self.action_type.value,
            "structure_type": self.structure_type.value if self.structure_type else None,
            "operator": self.operator,
            "operators": self.operators,
            "target": self.target,
            "condition": self.condition,
            "true_branch": self.true_branch,
            "false_branch": self.false_branch,
            "loop_count": self.loop_count,
            "position": self.position,
            "custom_prompt": self.custom_prompt,
            "raw_text": self.raw_text,
            "parse_error": self.parse_error,
            "reasoning": self.reasoning,
            "final_answer": self.final_answer
        }


class ActionParser:
    """XML 

     XML 
    """

    ACTION_PATTERN = re.compile(r'[<\[]action[>\]]\s*(add|delete|modify|set_prompt|finish)\s*[<\[]/action[>\]]', re.IGNORECASE)
    OPERATOR_PATTERN = re.compile(r'[<\[]operator[>\]]\s*([^<\[\]]+?)\s*[<\[]/operator[>\]]', re.IGNORECASE)
    OPERATORS_PATTERN = re.compile(r'[<\[]operators[>\]]\s*([^<\[\]]+?)\s*[<\[]/operators[>\]]', re.IGNORECASE)
    STRUCTURE_PATTERN = re.compile(r'[<\[]structure[>\]]\s*(parallel|conditional|loop)\s*[<\[]/structure[>\]]', re.IGNORECASE)
    TARGET_PATTERN = re.compile(r'[<\[]target[>\]]\s*([^<\[\]]+?)\s*[<\[]/target[>\]]', re.IGNORECASE)
    CONDITION_PATTERN = re.compile(r'[<\[]condition[>\]]\s*([^<\[\]]+?)\s*[<\[]/condition[>\]]', re.IGNORECASE)
    TRUE_PATTERN = re.compile(r'[<\[]true[>\]]\s*([^<\[\]]+?)\s*[<\[]/true[>\]]', re.IGNORECASE)
    FALSE_PATTERN = re.compile(r'[<\[]false[>\]]\s*([^<\[\]]+?)\s*[<\[]/false[>\]]', re.IGNORECASE)
    COUNT_PATTERN = re.compile(r'[<\[]count[>\]]\s*(\d+)\s*[<\[]/count[>\]]', re.IGNORECASE)
    POSITION_PATTERN = re.compile(r'[<\[]position[>\]]\s*(\d+)\s*[<\[]/position[>\]]', re.IGNORECASE)
    REASONING_PATTERN = re.compile(r'[<\[]reasoning[>\]]\s*(.*?)\s*[<\[]/reasoning[>\]]', re.IGNORECASE | re.DOTALL)

    ANSWER_PATTERN = re.compile(r'[<\[]answer[>\]]\s*(.*?)\s*[<\[]/answer[>\]]', re.IGNORECASE | re.DOTALL)
    THOUGHT_PATTERN = re.compile(r'[<\[]thought[>\]]\s*(.*?)\s*[<\[]/thought[>\]]', re.IGNORECASE | re.DOTALL)
    PROMPT_PATTERN = re.compile(r'[<\[]prompt[>\]]\s*(.*?)\s*[<\[]/prompt[>\]]', re.IGNORECASE | re.DOTALL)

    def __init__(self, strict: bool = False):
        """

        Args:
            strict:  ()
        """
        self.strict = strict

    # Pattern to match code blocks (```...```)
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)

    def _strip_code_blocks(self, text: str) -> str:
        """

        Args:
            text: 

        Returns:
        """
        return self.CODE_BLOCK_PATTERN.sub('', text)

    def parse(self, text: str) -> ParsedAction:
        """

        Args:
            text: 

        Returns:
            ParsedAction 
        """
        if not text or not text.strip():
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=text or "",
                parse_error="Empty input"
            )

        think_only_match = re.search(r'<think_only>(.*?)</think_only>', text, re.DOTALL)
        if think_only_match:
            thinking_snippet = think_only_match.group(1).strip()[:80]
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=text,
                parse_error=f"Think-Only: You output thinking without action. Please add <action> tag after your thinking.",
                reasoning=thinking_snippet
            )

        text_no_codeblocks = self._strip_code_blocks(text)

        reasoning = self._extract_tag(text_no_codeblocks, self.REASONING_PATTERN)
        if not reasoning:
            reasoning = self._extract_tag(text_no_codeblocks, self.THOUGHT_PATTERN)

        action_match = self.ACTION_PATTERN.search(text_no_codeblocks)
        if not action_match:
            if not self.strict:
                return self._fallback_parse(text, reasoning)
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=text,
                parse_error="No <action> tag found",
                reasoning=reasoning
            )

        action_str = action_match.group(1).lower()

        if action_str == "add":
            return self._parse_add(text_no_codeblocks, reasoning, text)
        elif action_str == "delete":
            return self._parse_delete(text_no_codeblocks, reasoning, text)
        elif action_str == "modify":
            return self._parse_modify(text_no_codeblocks, reasoning, text)
        elif action_str == "set_prompt":
            return self._parse_set_prompt(text_no_codeblocks, reasoning, text)
        elif action_str == "finish":

            final_answer = self._extract_tag(text_no_codeblocks, self.ANSWER_PATTERN)
            return ParsedAction(
                action_type=ActionType.FINISH,
                raw_text=text,
                reasoning=reasoning,
                final_answer=final_answer
            )
        else:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=text,
                parse_error=f"Unknown action: {action_str}",
                reasoning=reasoning
            )

    def _extract_tag(self, text: str, pattern: re.Pattern) -> Optional[str]:
        """"""
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    def _parse_operators_list(self, operators_str: str) -> List[str]:
        """"""
        operators = re.split(r'[,;\s]+', operators_str)
        return [op.strip() for op in operators if op.strip()]

    def _parse_add(self, text: str, reasoning: Optional[str], raw_text: str = None) -> ParsedAction:
        """ add """
        raw_text = raw_text or text
        structure_str = self._extract_tag(text, self.STRUCTURE_PATTERN)

        position_str = self._extract_tag(text, self.POSITION_PATTERN)
        position = int(position_str) if position_str else None

        if structure_str:
            structure_str = structure_str.lower()

            if structure_str == "parallel":
                return self._parse_add_parallel(text, reasoning, position, raw_text)
            elif structure_str == "conditional":
                return self._parse_add_conditional(text, reasoning, position, raw_text)
            elif structure_str == "loop":
                return self._parse_add_loop(text, reasoning, position, raw_text)
            else:
                return ParsedAction(
                    action_type=ActionType.INVALID,
                    raw_text=raw_text,
                    parse_error=f"Unknown structure: {structure_str}",
                    reasoning=reasoning
                )
        else:
            return self._parse_add_operator(text, reasoning, position, raw_text)

    def _parse_add_operator(self, text: str, reasoning: Optional[str],
                           position: Optional[int], raw_text: str = None) -> ParsedAction:
        """"""
        raw_text = raw_text or text
        operator = self._extract_tag(text, self.OPERATOR_PATTERN)
        custom_prompt = self._extract_tag(text, self.PROMPT_PATTERN)

        if not operator:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <operator> tag found for add action",
                reasoning=reasoning
            )

        return ParsedAction(
            action_type=ActionType.ADD,
            structure_type=StructureType.OPERATOR,
            operator=operator,
            position=position,
            custom_prompt=custom_prompt,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_add_parallel(self, text: str, reasoning: Optional[str],
                           position: Optional[int], raw_text: str = None) -> ParsedAction:
        """"""
        raw_text = raw_text or text
        operators_str = self._extract_tag(text, self.OPERATORS_PATTERN)

        if not operators_str:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <operators> tag found for parallel structure",
                reasoning=reasoning
            )

        operators = self._parse_operators_list(operators_str)

        if len(operators) < 2:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="Parallel structure requires at least 2 operators",
                reasoning=reasoning
            )

        return ParsedAction(
            action_type=ActionType.ADD,
            structure_type=StructureType.PARALLEL,
            operators=operators,
            position=position,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_add_conditional(self, text: str, reasoning: Optional[str],
                              position: Optional[int], raw_text: str = None) -> ParsedAction:
        """"""
        raw_text = raw_text or text
        condition = self._extract_tag(text, self.CONDITION_PATTERN)
        true_branch = self._extract_tag(text, self.TRUE_PATTERN)
        false_branch = self._extract_tag(text, self.FALSE_PATTERN)

        if not condition:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <condition> tag found for conditional structure",
                reasoning=reasoning
            )

        if not true_branch:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <true> tag found for conditional structure",
                reasoning=reasoning
            )

        if false_branch and false_branch.lower() == "done":
            false_branch = None

        return ParsedAction(
            action_type=ActionType.ADD,
            structure_type=StructureType.CONDITIONAL,
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
            position=position,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_add_loop(self, text: str, reasoning: Optional[str],
                       position: Optional[int], raw_text: str = None) -> ParsedAction:
        """"""
        raw_text = raw_text or text
        operators_str = self._extract_tag(text, self.OPERATORS_PATTERN)
        count_str = self._extract_tag(text, self.COUNT_PATTERN)

        if not operators_str:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <operators> tag found for loop structure",
                reasoning=reasoning
            )

        operators = self._parse_operators_list(operators_str)

        if not operators:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="Loop structure requires at least 1 operator",
                reasoning=reasoning
            )

        loop_count = int(count_str) if count_str else 3
        loop_count = max(1, min(10, loop_count))

        return ParsedAction(
            action_type=ActionType.ADD,
            structure_type=StructureType.LOOP,
            operators=operators,
            loop_count=loop_count,
            position=position,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_delete(self, text: str, reasoning: Optional[str], raw_text: str = None) -> ParsedAction:
        """ delete """
        raw_text = raw_text or text
        target = self._extract_tag(text, self.TARGET_PATTERN)

        if not target:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <target> tag found for delete action",
                reasoning=reasoning
            )

        return ParsedAction(
            action_type=ActionType.DELETE,
            target=target,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_modify(self, text: str, reasoning: Optional[str], raw_text: str = None) -> ParsedAction:
        """ modify """
        raw_text = raw_text or text
        target = self._extract_tag(text, self.TARGET_PATTERN)
        operator = self._extract_tag(text, self.OPERATOR_PATTERN)

        if not target:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <target> tag found for modify action",
                reasoning=reasoning
            )

        if not operator:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <operator> tag found for modify action",
                reasoning=reasoning
            )

        return ParsedAction(
            action_type=ActionType.MODIFY,
            target=target,
            operator=operator,
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _parse_set_prompt(self, text: str, reasoning: Optional[str], raw_text: str = None) -> ParsedAction:
        """ set_prompt  ()

         operator  operator  prompt

        :
            1) / promptBUILDING :
               <action>set_prompt</action><target>node_ID</target><prompt>...</prompt>
            2) AWAITING_PROMPT  <prompt> :
               <action>set_prompt</action><prompt>...</prompt>
        """
        raw_text = raw_text or text
        target = self._extract_tag(text, self.TARGET_PATTERN)
        custom_prompt = self._extract_tag(text, self.PROMPT_PATTERN)

        if not custom_prompt:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="No <prompt> tag found for set_prompt action. You must provide a prompt.",
                reasoning=reasoning
            )

        if not custom_prompt.strip():
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=raw_text,
                parse_error="Empty prompt content. Please provide a meaningful prompt.",
                reasoning=reasoning
            )

        return ParsedAction(
            action_type=ActionType.SET_PROMPT,
            target=target.strip() if target else None,
            custom_prompt=custom_prompt.strip(),
            raw_text=raw_text,
            reasoning=reasoning
        )

    def _fallback_parse(self, text: str, reasoning: Optional[str]) -> ParsedAction:
        """ ()

        -  → FINISH
        -  → ADD operator

         "complete"/"done" 
        """
        text_for_search = text
        text_for_search = re.sub(r'<think>.*?</think>', '', text_for_search, flags=re.DOTALL | re.IGNORECASE)
        text_for_search = re.sub(r'<thinking>.*?</thinking>', '', text_for_search, flags=re.DOTALL | re.IGNORECASE)
        text_for_search = re.sub(r'<reasoning>.*?</reasoning>', '', text_for_search, flags=re.DOTALL | re.IGNORECASE)

        text_lower = text_for_search.lower()

        finish_patterns = [
            r'\blet\'?s?\s+finish\b',           # let's finish / lets finish
            r'\bi\s+will\s+finish\b',           # i will finish
            r'\bfinish\s+(the\s+)?workflow\b',  # finish workflow / finish the workflow
            r'\bworkflow\s+(is\s+)?complete\b', # workflow complete / workflow is complete
            r'\btime\s+to\s+finish\b',          # time to finish
            r'\bnow\s+finish\b',                # now finish
            r'\bready\s+to\s+finish\b',         # ready to finish
        ]

        for pattern in finish_patterns:
            if re.search(pattern, text_lower):
                return ParsedAction(
                    action_type=ActionType.FINISH,
                    raw_text=text,
                    reasoning=reasoning,
                    parse_error="Fallback: inferred FINISH from explicit finish phrase"
                )

        text_for_search = re.sub(r'<reasoning>.*?</reasoning>', '', text_for_search, flags=re.DOTALL | re.IGNORECASE)

        try:
            from .workflow_graph import VALID_OPERATORS
        except ImportError:
            VALID_OPERATORS = {
                'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
                'Programmer', 'Test', 'Format',
                'Review', 'Revise', 'ScEnsemble', 'MdEnsemble',
                'Decompose', 'Verify', 'Plan', 'Aggregate'
            }

        found_operators = []
        for op in VALID_OPERATORS:
            if re.search(rf'\b{op}\b', text_for_search, re.IGNORECASE):
                found_operators.append(op)

        if len(found_operators) == 1:
            return ParsedAction(
                action_type=ActionType.ADD,
                structure_type=StructureType.OPERATOR,
                operator=found_operators[0],
                raw_text=text,
                reasoning=reasoning,
                parse_error="Fallback: inferred ADD from operator mention"
            )
        elif len(found_operators) > 1:
            return ParsedAction(
                action_type=ActionType.INVALID,
                raw_text=text,
                reasoning=reasoning,
                parse_error=f"Found multiple operators ({', '.join(found_operators)}) but no clear structure. Please use explicit XML format."
            )

        return ParsedAction(
            action_type=ActionType.INVALID,
            raw_text=text,
            parse_error="Could not parse action from text",
            reasoning=reasoning
        )

    def format_action_prompt(self) -> str:
        """ ( system prompt)"""
        return '''You can build a workflow step by step using these XML commands:

1. Add a single operator:
   <action>add</action><operator>OPERATOR_NAME</operator>

2. Add parallel operators (execute simultaneously):
   <action>add</action><structure>parallel</structure><operators>OP1,OP2,OP3</operators>

3. Add conditional branch (if-else):
   <action>add</action><structure>conditional</structure>
   <condition>CONDITION_OP</condition><true>TRUE_OP</true><false>FALSE_OP or done</false>

4. Add loop structure:
   <action>add</action><structure>loop</structure><operators>OP1,OP2</operators><count>3</count>

5. Delete a node:
   <action>delete</action><target>NODE_ID</target>

6. Modify a node:
   <action>modify</action><target>NODE_ID</target><operator>NEW_OPERATOR</operator>

7. Finish building:
   <action>finish</action>

Available operators: Custom, AnswerGenerate, Programmer, Test, Review, Revise,
ScEnsemble, MdEnsemble, Format, Decompose, Verify, Plan, Aggregate

You can optionally include your reasoning:
<reasoning>Your thought process here</reasoning>
'''


def parse_action(text: str, strict: bool = False) -> ParsedAction:
    """

    Args:
        text: 
        strict: 

    Returns:
        ParsedAction 
    """
    parser = ActionParser(strict=strict)
    return parser.parse(text)


def extract_actions(text: str) -> List[ParsedAction]:
    """ ()

    Args:
        text: 

    Returns:
        ParsedAction 
    """
    parser = ActionParser(strict=False)
    actions = []

    parts = re.split(r'(?=<action>)', text)

    for part in parts:
        part = part.strip()
        if not part or '<action>' not in part:
            continue
        action = parser.parse(part)
        if action.action_type != ActionType.INVALID:
            actions.append(action)

    return actions


if __name__ == "__main__":
    print("=" * 60)
    print("ActionParser ")
    print("=" * 60)

    parser = ActionParser()

    test_cases = [
        "<action>add</action><operator>Custom</operator>",

        "<action>add</action><structure>parallel</structure><operators>Custom, Programmer, Review</operators>",

        "<action>add</action><structure>conditional</structure><condition>Review</condition><true>Revise</true><false>done</false>",

        "<action>add</action><structure>loop</structure><operators>Custom, Review</operators><count>3</count>",

        "<action>delete</action><target>node_3</target>",

        "<action>modify</action><target>node_2</target><operator>Programmer</operator>",

        "<action>finish</action>",

        "<reasoning>This problem needs code verification, so I'll add Programmer.</reasoning><action>add</action><operator>Programmer</operator>",

        "I think we should add Custom operator here",

        "The workflow looks complete, let's finish.",

        "This is just random text",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. : {test[:60]}..." if len(test) > 60 else f"\n{i}. : {test}")
        result = parser.parse(test)
        print(f"   : {result.action_type.value}")
        if result.structure_type:
            print(f"   : {result.structure_type.value}")
        if result.operator:
            print(f"   : {result.operator}")
        if result.operators:
            print(f"   : {result.operators}")
        if result.target:
            print(f"   : {result.target}")
        if result.condition:
            print(f"   : {result.condition}")
        if result.true_branch:
            print(f"   True: {result.true_branch}")
        if result.false_branch:
            print(f"   False: {result.false_branch}")
        if result.loop_count != 3 or result.structure_type == StructureType.LOOP:
            print(f"   : {result.loop_count}")
        if result.reasoning:
            print(f"   : {result.reasoning[:50]}...")
        if result.parse_error:
            print(f"   : {result.parse_error}")
        print(f"   : {result.is_valid()}")

    print("\n" + "=" * 60)
    print("!")
