#!/usr/bin/env python3
"""
Response Standardizer - 
"""
from typing import Dict, Any, Optional


class ResponseStandardizer:
    """

    :
    {
        'success': bool,
        'content': str,
        'metadata': dict,
        'error': Optional[str]
    }
    """

    @staticmethod
    def standardize(raw_response: Any, operator_type: str) -> Dict[str, Any]:
        """

        Args:
            raw_response: 
            operator_type: 

        Returns:
        """
        if raw_response is None:
            return {
                'success': False,
                'content': '',
                'metadata': {},
                'error': 'Operator returned None'
            }

        if isinstance(raw_response, str):
            return {
                'success': True,
                'content': raw_response,
                'metadata': {},
                'error': None
            }

        if not isinstance(raw_response, dict):
            return {
                'success': True,
                'content': str(raw_response),
                'metadata': {},
                'error': None
            }

        if operator_type == 'Custom':
            return ResponseStandardizer._standardize_custom(raw_response)
        elif operator_type == 'AnswerGenerate':
            return ResponseStandardizer._standardize_answer_generate(raw_response)
        elif operator_type == 'Programmer':
            return ResponseStandardizer._standardize_programmer(raw_response)
        elif operator_type == 'Test':
            return ResponseStandardizer._standardize_test(raw_response)
        elif operator_type == 'Review':
            return ResponseStandardizer._standardize_review(raw_response)
        elif operator_type == 'Revise':
            return ResponseStandardizer._standardize_revise(raw_response)
        elif operator_type == 'ScEnsemble':
            return ResponseStandardizer._standardize_ensemble(raw_response)
        elif operator_type == 'Format':
            return ResponseStandardizer._standardize_format(raw_response)
        else:
            return ResponseStandardizer._standardize_generic(raw_response)

    @staticmethod
    def _standardize_custom(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('answer', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_answer_generate(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('answer', ''),
            'metadata': {
                'thought': resp.get('thought', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_programmer(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('output', ''),
            'metadata': {
                'code': resp.get('code', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_test(resp: Dict) -> Dict:
        return {
            'success': resp.get('result', False),
            'content': resp.get('solution', ''),
            'metadata': {
                'test_result': resp.get('result', False),
                'original': resp
            },
            'error': None if resp.get('result', False) else 'Test failed'
        }

    @staticmethod
    def _standardize_review(resp: Dict) -> Dict:
        feedback = (
            resp.get('feedback') or
            resp.get('review_result') or
            resp.get('review') or
            'Review completed'
        )

        review_result = resp.get('review_result', True)
        if isinstance(review_result, str):
            review_result = 'pass' in review_result.lower() or 'success' in review_result.lower()

        return {
            'success': True,
            'content': feedback,
            'metadata': {
                'review_result': review_result,
                'feedback': feedback,
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_revise(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('solution', resp.get('code', '')),
            'metadata': {
                'solution': resp.get('solution', ''),
                'original': resp
            },
            'error': None
        }

    @staticmethod
    def _standardize_ensemble(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('solution', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_format(resp: Dict) -> Dict:
        return {
            'success': True,
            'content': resp.get('response', resp.get('formatted', '')),
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def _standardize_generic(resp: Dict) -> Dict:
        """ - """
        content_fields = ['response', 'answer', 'solution', 'output', 'result', 'code']
        content = ''

        for field in content_fields:
            if field in resp:
                content = resp[field]
                break

        if not content:
            content = str(resp)

        return {
            'success': True,
            'content': content,
            'metadata': {'original': resp},
            'error': None
        }

    @staticmethod
    def safe_get(data: Dict, *keys, default='') -> Any:
        """

        Args:
            data: 
            *keys: 
            default: 

        Returns:
        """
        for key in keys:
            value = data.get(key)
            if value is not None and value != '':
                return value
        return default
