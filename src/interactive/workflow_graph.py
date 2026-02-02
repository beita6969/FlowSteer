# -*- coding: utf-8 -*-
"""
Interactive Workflow Building - WorkflowGraph Data Structure
=============================================================
 DSL

DSL :
- : A -> B -> C
- : [A, B, C]
- : A ? B : C
- : (A) * 3  (A -> B) * 3

: Claude Code
: 2024-12-09
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid


class NodeType(Enum):
    """"""
    OPERATOR = "operator"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"


VALID_OPERATORS = {
    'Custom', 'AnswerGenerate', 'CustomCodeGenerate',
    'Programmer', 'Test', 'Format',
    'Review', 'Revise', 'ScEnsemble', 'MdEnsemble',
    'Decompose', 'Verify', 'Plan', 'Aggregate'
}


@dataclass
class WorkflowNode:
    """

    :
    1. OPERATOR:  (e.g., "Custom", "Programmer")
    2. PARALLEL:  (children )
    3. CONDITIONAL:  (condition , true_branch/false_branch )
    4. LOOP:  (children , loop_count )
    """
    id: str
    node_type: NodeType
    operator: Optional[str] = None
    children: List['WorkflowNode'] = field(default_factory=list)
    condition: Optional['WorkflowNode'] = None
    true_branch: Optional['WorkflowNode'] = None
    false_branch: Optional['WorkflowNode'] = None
    loop_count: int = 1
    custom_prompt: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = f"node_{uuid.uuid4().hex[:8]}"

    def to_dsl(self) -> str:
        """ DSL """
        if self.node_type == NodeType.OPERATOR:
            return self.operator or ""

        elif self.node_type == NodeType.PARALLEL:
            if not self.children:
                return ""
            children_dsl = ", ".join(child.to_dsl() for child in self.children)
            return f"[{children_dsl}]"

        elif self.node_type == NodeType.CONDITIONAL:
            if not self.condition or not self.true_branch:
                return ""
            cond_dsl = self.condition.to_dsl()
            true_dsl = self.true_branch.to_dsl()
            false_dsl = self.false_branch.to_dsl() if self.false_branch else "done"
            return f"{cond_dsl} ? {true_dsl} : {false_dsl}"

        elif self.node_type == NodeType.LOOP:
            if not self.children:
                return ""
            if len(self.children) == 1:
                body_dsl = self.children[0].to_dsl()
            else:
                body_dsl = " -> ".join(child.to_dsl() for child in self.children)
            return f"({body_dsl}) * {self.loop_count}"

        return ""

    def get_operators(self) -> List[str]:
        """ operator"""
        operators = []

        if self.node_type == NodeType.OPERATOR and self.operator:
            operators.append(self.operator)

        for child in self.children:
            operators.extend(child.get_operators())

        if self.condition:
            operators.extend(self.condition.get_operators())
        if self.true_branch:
            operators.extend(self.true_branch.get_operators())
        if self.false_branch:
            operators.extend(self.false_branch.get_operators())

        return operators

    def clone(self) -> 'WorkflowNode':
        """"""
        new_node = WorkflowNode(
            id=f"node_{uuid.uuid4().hex[:8]}",
            node_type=self.node_type,
            operator=self.operator,
            loop_count=self.loop_count,
            custom_prompt=self.custom_prompt
        )
        new_node.children = [child.clone() for child in self.children]
        new_node.condition = self.condition.clone() if self.condition else None
        new_node.true_branch = self.true_branch.clone() if self.true_branch else None
        new_node.false_branch = self.false_branch.clone() if self.false_branch else None
        return new_node


class WorkflowGraph:
    """

    :
    - add_operator(): 
    - add_parallel(): 
    - add_conditional(): 
    - add_loop(): 
    - remove_node(): 
    - modify_node(): 
    - to_dsl():  DSL 
    """

    def __init__(self):
        self.nodes: List[WorkflowNode] = []
        self._node_map: Dict[str, WorkflowNode] = {}

    def _register_node(self, node: WorkflowNode) -> None:
        """"""
        self._node_map[node.id] = node
        for child in node.children:
            self._register_node(child)
        if node.condition:
            self._register_node(node.condition)
        if node.true_branch:
            self._register_node(node.true_branch)
        if node.false_branch:
            self._register_node(node.false_branch)

    def _unregister_node(self, node: WorkflowNode) -> None:
        """"""
        if node.id in self._node_map:
            del self._node_map[node.id]
        for child in node.children:
            self._unregister_node(child)
        if node.condition:
            self._unregister_node(node.condition)
        if node.true_branch:
            self._unregister_node(node.true_branch)
        if node.false_branch:
            self._unregister_node(node.false_branch)

    def add_operator(self, operator: str, position: Optional[int] = None) -> Dict[str, Any]:
        """

        Args:
            operator: 
            position:  (None )

        Returns:
            {"success": bool, "node_id": str, "message": str}
        """
        if operator not in VALID_OPERATORS:
            return {
                "success": False,
                "node_id": None,
                "message": f"Invalid operator: {operator}. Valid operators: {', '.join(sorted(VALID_OPERATORS))}"
            }

        node = WorkflowNode(
            id=f"node_{len(self.nodes) + 1}",
            node_type=NodeType.OPERATOR,
            operator=operator
        )

        if position is None or position >= len(self.nodes):
            self.nodes.append(node)
            pos_desc = "at end"
        else:
            position = max(0, position)
            self.nodes.insert(position, node)
            pos_desc = f"at position {position}"

        self._register_node(node)

        return {
            "success": True,
            "node_id": node.id,
            "message": f"Added {operator} {pos_desc}"
        }

    def add_parallel(self, operators: List[str], position: Optional[int] = None) -> Dict[str, Any]:
        """

        Args:
            operators: 
            position: 

        Returns:
            {"success": bool, "node_id": str, "message": str}
        """
        FORBIDDEN_IN_PARALLEL = {'ScEnsemble', 'Aggregate', 'MdEnsemble'}
        forbidden_found = [op for op in operators if op in FORBIDDEN_IN_PARALLEL]
        if forbidden_found:
            return {
                "success": False,
                "node_id": None,
                "message": f"{', '.join(forbidden_found)} cannot be inside parallel. Add them AFTER the parallel block."
            }

        if not operators or len(operators) < 2:
            return {
                "success": False,
                "node_id": None,
                "message": "Parallel structure requires at least 2 operators"
            }

        invalid_ops = [op for op in operators if op not in VALID_OPERATORS]
        if invalid_ops:
            return {
                "success": False,
                "node_id": None,
                "message": f"Invalid operators: {', '.join(invalid_ops)}"
            }

        children = []
        for i, op in enumerate(operators):
            child = WorkflowNode(
                id=f"node_{len(self.nodes) + 1}_p{i}",
                node_type=NodeType.OPERATOR,
                operator=op
            )
            children.append(child)

        node = WorkflowNode(
            id=f"node_{len(self.nodes) + 1}",
            node_type=NodeType.PARALLEL,
            children=children
        )

        if position is None or position >= len(self.nodes):
            self.nodes.append(node)
        else:
            self.nodes.insert(max(0, position), node)

        self._register_node(node)

        return {
            "success": True,
            "node_id": node.id,
            "message": f"Added parallel [{', '.join(operators)}]"
        }

    def add_conditional(self, condition_op: str, true_op: str,
                       false_op: Optional[str] = None,
                       position: Optional[int] = None) -> Dict[str, Any]:
        """

        Args:
            condition_op:  ( Review)
            true_op: 
            false_op:  (None  "done")
            position: 

        Returns:
            {"success": bool, "node_id": str, "message": str}
        """
        ops_to_check = [condition_op, true_op]
        if false_op:
            ops_to_check.append(false_op)

        invalid_ops = [op for op in ops_to_check if op not in VALID_OPERATORS]
        if invalid_ops:
            return {
                "success": False,
                "node_id": None,
                "message": f"Invalid operators: {', '.join(invalid_ops)}"
            }

        base_id = f"node_{len(self.nodes) + 1}"

        condition_node = WorkflowNode(
            id=f"{base_id}_cond",
            node_type=NodeType.OPERATOR,
            operator=condition_op
        )

        true_node = WorkflowNode(
            id=f"{base_id}_true",
            node_type=NodeType.OPERATOR,
            operator=true_op
        )

        false_node = None
        if false_op:
            false_node = WorkflowNode(
                id=f"{base_id}_false",
                node_type=NodeType.OPERATOR,
                operator=false_op
            )

        node = WorkflowNode(
            id=base_id,
            node_type=NodeType.CONDITIONAL,
            condition=condition_node,
            true_branch=true_node,
            false_branch=false_node
        )

        if position is None or position >= len(self.nodes):
            self.nodes.append(node)
        else:
            self.nodes.insert(max(0, position), node)

        self._register_node(node)

        false_desc = false_op if false_op else "done"
        return {
            "success": True,
            "node_id": node.id,
            "message": f"Added conditional: {condition_op} ? {true_op} : {false_desc}"
        }

    def add_loop(self, operators: List[str], count: int = 3,
                 position: Optional[int] = None) -> Dict[str, Any]:
        """

        Args:
            operators: 
            count: 
            position: 

        Returns:
            {"success": bool, "node_id": str, "message": str}
        """
        if not operators:
            return {
                "success": False,
                "node_id": None,
                "message": "Loop requires at least 1 operator"
            }

        if count < 1 or count > 10:
            return {
                "success": False,
                "node_id": None,
                "message": "Loop count must be between 1 and 10"
            }

        invalid_ops = [op for op in operators if op not in VALID_OPERATORS]
        if invalid_ops:
            return {
                "success": False,
                "node_id": None,
                "message": f"Invalid operators: {', '.join(invalid_ops)}"
            }

        base_id = f"node_{len(self.nodes) + 1}"

        children = []
        for i, op in enumerate(operators):
            child = WorkflowNode(
                id=f"{base_id}_l{i}",
                node_type=NodeType.OPERATOR,
                operator=op
            )
            children.append(child)

        node = WorkflowNode(
            id=base_id,
            node_type=NodeType.LOOP,
            children=children,
            loop_count=count
        )

        if position is None or position >= len(self.nodes):
            self.nodes.append(node)
        else:
            self.nodes.insert(max(0, position), node)

        self._register_node(node)

        if len(operators) == 1:
            body_desc = operators[0]
        else:
            body_desc = " -> ".join(operators)

        return {
            "success": True,
            "node_id": node.id,
            "message": f"Added loop: ({body_desc}) * {count}"
        }

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        """

        Args:
            node_id:  ID

        Returns:
            {"success": bool, "message": str}
        """
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                self._unregister_node(node)
                self.nodes.pop(i)
                return {
                    "success": True,
                    "message": f"Removed node {node_id}"
                }

        if node_id in self._node_map:
            return {
                "success": False,
                "message": f"Cannot remove child node {node_id}. Remove parent node instead."
            }

        return {
            "success": False,
            "message": f"Node {node_id} not found"
        }

    def modify_node(self, node_id: str, new_operator: str) -> Dict[str, Any]:
        """ operator

        Args:
            node_id:  ID
            new_operator: 

        Returns:
            {"success": bool, "message": str}
        """
        if new_operator not in VALID_OPERATORS:
            return {
                "success": False,
                "message": f"Invalid operator: {new_operator}"
            }

        if node_id not in self._node_map:
            return {
                "success": False,
                "message": f"Node {node_id} not found"
            }

        node = self._node_map[node_id]

        if node.node_type != NodeType.OPERATOR:
            return {
                "success": False,
                "message": f"Cannot modify non-operator node {node_id}"
            }

        old_op = node.operator
        node.operator = new_operator

        return {
            "success": True,
            "message": f"Modified {node_id}: {old_op} -> {new_operator}"
        }

    def set_node_prompt(self, node_id: str, custom_prompt: str) -> Dict[str, Any]:
        """ prompt ()

        Args:
            node_id:  prompt  ID
            custom_prompt:  prompt 

        Returns:
            {"success": bool, "node_id": str, "operator": str, "message": str}
        """
        if "PREVIOUS_ERROR" in str(custom_prompt):
            print(f"[DEBUG] set_node_prompt: node_id={node_id}, has_error=True", flush=True)

        if node_id not in self._node_map:
            return {
                "success": False,
                "node_id": node_id,
                "operator": None,
                "message": f"Node {node_id} not found"
            }

        node = self._node_map[node_id]

        if node.node_type != NodeType.OPERATOR:
            return {
                "success": False,
                "node_id": node_id,
                "operator": None,
                "message": f"Cannot set prompt for non-operator node {node_id}"
            }

        node.custom_prompt = custom_prompt

        return {
            "success": True,
            "node_id": node_id,
            "operator": node.operator,
            "message": f"Set custom prompt for {node.operator} ({node_id})"
        }

    def get_last_added_node_id(self) -> Optional[str]:
        """ ID

        : ADD  ID  SET_PROMPT

        Returns:
             ID,  None
        """
        if not self.nodes:
            return None
        return self.nodes[-1].id

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """

        Args:
            node_id:  ID

        Returns:
             None
        """
        if node_id not in self._node_map:
            return None

        node = self._node_map[node_id]
        return {
            "id": node.id,
            "type": node.node_type.value,
            "operator": node.operator,
            "custom_prompt": node.custom_prompt,
            "dsl": node.to_dsl()
        }

    def get_all_prompts(self) -> Dict[str, Dict[str, str]]:
        """ custom_prompt 

        Returns:
            {node_id: {"operator": str, "prompt": str}, ...}
        """
        prompts = {}
        for node_id, node in self._node_map.items():
            if node.node_type == NodeType.OPERATOR and node.custom_prompt:
                prompts[node_id] = {
                    "operator": node.operator,
                    "prompt": node.custom_prompt
                }
                if "PREVIOUS_ERROR" in str(node.custom_prompt):
                    print(f"[DEBUG] get_all_prompts: node_id={node_id}, has_error=True", flush=True)
        return prompts

    def to_dsl(self) -> str:
        """ DSL 

        Returns:
            DSL  (e.g., "Plan -> [Custom, Programmer] -> ScEnsemble -> Review ? Revise : done")
        """
        if not self.nodes:
            return ""

        return " -> ".join(node.to_dsl() for node in self.nodes)

    def get_all_operators(self) -> List[str]:
        """ operator"""
        operators = []
        for node in self.nodes:
            operators.extend(node.get_operators())
        return operators

    def get_unique_operators(self) -> set:
        """ operator """
        return set(self.get_all_operators())

    def get_statistics(self) -> Dict[str, Any]:
        """"""
        all_ops = self.get_all_operators()
        unique_ops = set(all_ops)

        has_parallel = any(n.node_type == NodeType.PARALLEL for n in self.nodes)
        has_conditional = any(n.node_type == NodeType.CONDITIONAL for n in self.nodes)
        has_loop = any(n.node_type == NodeType.LOOP for n in self.nodes)

        def check_nested(nodes: List[WorkflowNode]) -> Dict[str, bool]:
            result = {"parallel": False, "conditional": False, "loop": False}
            for n in nodes:
                if n.node_type == NodeType.PARALLEL:
                    result["parallel"] = True
                elif n.node_type == NodeType.CONDITIONAL:
                    result["conditional"] = True
                elif n.node_type == NodeType.LOOP:
                    result["loop"] = True
                    nested = check_nested(n.children)
                    for k, v in nested.items():
                        result[k] = result[k] or v
            return result

        nested = check_nested(self.nodes)

        return {
            "total_operators": len(all_ops),
            "unique_types": len(unique_ops),
            "operator_list": all_ops,
            "unique_operator_set": unique_ops,
            "has_parallel": has_parallel or nested["parallel"],
            "has_conditional": has_conditional or nested["conditional"],
            "has_loop": has_loop or nested["loop"],
            "node_count": len(self.nodes),
            "dsl_length": len(self.to_dsl())
        }

    def get_node_list(self) -> List[Dict[str, Any]]:
        """"""
        result = []
        for i, node in enumerate(self.nodes):
            result.append({
                "index": i,
                "id": node.id,
                "type": node.node_type.value,
                "dsl": node.to_dsl()
            })
        return result

    def clear(self) -> None:
        """"""
        self.nodes.clear()
        self._node_map.clear()

    def clone(self) -> 'WorkflowGraph':
        """"""
        new_graph = WorkflowGraph()
        for node in self.nodes:
            cloned = node.clone()
            new_graph.nodes.append(cloned)
            new_graph._register_node(cloned)
        return new_graph

    def is_empty(self) -> bool:
        """"""
        return len(self.nodes) == 0

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> str:
        return self.to_dsl() or "(empty workflow)"

    def __repr__(self) -> str:
        return f"WorkflowGraph(nodes={len(self.nodes)}, dsl='{self.to_dsl()}')"


# ============================================
# ============================================

def create_workflow_from_dsl(dsl: str) -> Optional[WorkflowGraph]:
    """ DSL  WorkflowGraph ()

    Note: 
     DSL  vllm_workflow_generator.py  WorkflowDSLParser

    Args:
        dsl: DSL 

    Returns:
        WorkflowGraph  None ()
    """
    if not dsl or not dsl.strip():
        return None

    graph = WorkflowGraph()

    parts = [p.strip() for p in dsl.split("->")]

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("[") and part.endswith("]"):
            operators = [op.strip() for op in part[1:-1].split(",")]
            result = graph.add_parallel(operators)
            if not result["success"]:
                return None
        elif part.startswith("(") and "*" in part:
            loop_match = part.split("*")
            if len(loop_match) == 2:
                body = loop_match[0].strip().strip("()")
                try:
                    count = int(loop_match[1].strip())
                except ValueError:
                    count = 3
                body_ops = [op.strip() for op in body.split("->")]
                result = graph.add_loop(body_ops, count)
                if not result["success"]:
                    return None
        elif "?" in part and ":" in part:
            q_idx = part.index("?")
            c_idx = part.index(":")
            cond = part[:q_idx].strip()
            true_op = part[q_idx+1:c_idx].strip()
            false_op = part[c_idx+1:].strip()
            if false_op.lower() == "done":
                false_op = None
            result = graph.add_conditional(cond, true_op, false_op)
            if not result["success"]:
                return None
        elif part in VALID_OPERATORS:
            result = graph.add_operator(part)
            if not result["success"]:
                return None

    return graph if not graph.is_empty() else None


if __name__ == "__main__":
    print("=" * 60)
    print("WorkflowGraph ")
    print("=" * 60)

    graph = WorkflowGraph()
    print(f"\n1. : {graph}")

    result = graph.add_operator("Plan")
    print(f"\n2.  Plan: {result}")
    print(f"   DSL: {graph.to_dsl()}")

    result = graph.add_parallel(["Custom", "Programmer", "Custom"])
    print(f"\n3. : {result}")
    print(f"   DSL: {graph.to_dsl()}")

    result = graph.add_operator("ScEnsemble")
    print(f"\n4.  ScEnsemble: {result}")
    print(f"   DSL: {graph.to_dsl()}")

    result = graph.add_conditional("Review", "Revise", None)
    print(f"\n5. : {result}")
    print(f"   DSL: {graph.to_dsl()}")

    stats = graph.get_statistics()
    print(f"\n6. :")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    print(f"\n7. :")
    for node_info in graph.get_node_list():
        print(f"   {node_info}")

    graph2 = WorkflowGraph()
    graph2.add_operator("Custom")
    graph2.add_loop(["Review", "Revise"], count=3)
    print(f"\n8. : {graph2.to_dsl()}")

    result = graph.remove_node("node_3")
    print(f"\n9.  node_3: {result}")
    print(f"   DSL: {graph.to_dsl()}")

    result = graph.modify_node("node_2_p0", "Programmer")
    print(f"\n10.  node_2_p0: {result}")
    print(f"    DSL: {graph.to_dsl()}")

    print("\n" + "=" * 60)
    print("!")
