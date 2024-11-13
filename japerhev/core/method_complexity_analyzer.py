from typing import List
import re

class MethodChangeComplexityAnalyzer:
    """Analyze the complexity of Java method-level changes based on diff output."""
    
    def __init__(self):
        self.weights = {
            'base_change': 1.0,      
            'structural': {
                'control_flow': 2.5,  # if, else, loops
                'method_sig': 2.0,    # Method signature changes
                'exception': 2.0,     # Exception handling
            },
            'scope': {
                'variable_intro': 1.5,  # New variable introduction
                'type_change': 1.5,     # Type system changes
            },
            'size': {
                'loc_modified': 1.0,    # Per modified line
                'chunk_size': 1.2,      # Size of continuous changes
            }
        }
    
    def _parse_diff_lines(self, diff_text: str) -> tuple[List[str], List[str]]:
        """Parse diff text into old and new lines.
        
        :param diff_text: Raw diff text
        :type diff_text: str
        :return: Tuple of old and new lines
        :rtype: tuple[List[str], List[str]]
        """
        old_lines = []
        new_lines = []
        for line in diff_text.splitlines():
            # Match the specific format with varying spaces
            match = re.match(r'\s*\d*\s*(-|\+)?\s*(\d*)?\s*([^-+].*)?$', line)
            if match:
                change_type = match.group(1)  # '-' or '+' or None
                content = match.group(3)
                
                if content:  # Only process if there's actual content
                    if change_type == '-':
                        old_lines.append(content.strip())
                    elif change_type == '+':
                        new_lines.append(content.strip())
        
        return old_lines, new_lines
    
    def _calculate_structural_complexity(self, old_lines: List[str], new_lines: List[str]) -> float:
        """Calculate structural complexity changes.
        
        :param old_lines: List of old lines
        :type old_lines: List[str]
        :param new_lines: List of new lines
        :type new_lines: List[str]
        :return: Structural complexity score
        :rtype: float
        """
        score = 0.0
        
        # Control flow changes
        control_patterns = r'\b(if|else|for|while|do|switch)\b'
        old_control = len(re.findall(control_patterns, ' '.join(old_lines)))
        new_control = len(re.findall(control_patterns, ' '.join(new_lines)))
        if old_control != new_control:
            score += self.weights['structural']['control_flow'] * abs(new_control - old_control)
        
        return score
    
    def _calculate_scope_complexity(self, old_lines: List[str], new_lines: List[str]) -> float:
        """Calculate scope-related complexity changes.
        
        :param old_lines: List of old lines
        :type old_lines: List[str]
        :param new_lines: List of new lines
        :type new_lines: List[str]
        :return: Scope complexity score
        :rtype: float
        """
        score = 0.0
        
        # Variable introductions (looking for new variable declarations)
        var_pattern = r'\b\w+\s+\w+\s*='
        old_vars = set(re.findall(var_pattern, ' '.join(old_lines)))
        new_vars = set(re.findall(var_pattern, ' '.join(new_lines)))
        new_var_count = len(new_vars - old_vars)
        if new_var_count > 0:
            score += self.weights['scope']['variable_intro'] * new_var_count
        
        return score
    
    def _calculate_size_complexity(self, old_lines: List[str], new_lines: List[str]) -> float:
        """Calculate size-based complexity.
        
        :param old_lines: List of old lines
        :type old_lines: List[str]
        :param new_lines: List of new lines
        :type new_lines: List[str]
        :return: Size complexity score
        :rtype: float
        """
        score = 0.0
        
        # Count total modified lines
        total_changes = len(old_lines) + len(new_lines)
        score += total_changes * self.weights['size']['loc_modified']
        
        # Add complexity for consecutive changes
        if total_changes > 1:
            score += (total_changes - 1) * self.weights['size']['chunk_size']
        
        return score
    
    def calculate_complexity(self, diff_text: str) -> float:
        """Calculate overall change complexity.
        
        :param diff_text: Raw diff text
        :type diff_text: str
        :return: Complexity score
        :rtype: float
        """
        if not diff_text.strip():
            return 0.0
        
        # Parse the diff
        old_lines, new_lines = self._parse_diff_lines(diff_text)
        
        if not old_lines and not new_lines:
            return 0.0
            
        # Start with base complexity
        total_score = self.weights['base_change']
        
        # Add structural complexity
        total_score += self._calculate_structural_complexity(old_lines, new_lines)
        
        # Add scope complexity
        total_score += self._calculate_scope_complexity(old_lines, new_lines)
        
        # Add size complexity
        total_score += self._calculate_size_complexity(old_lines, new_lines)
        
        return round(total_score, 2)