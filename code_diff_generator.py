from difflib import SequenceMatcher
from typing import Optional

class CodeDiffGenerator:
    """Generate a formatted code diff between two pieces of code."""
    
    def __init__(self, context_lines: int = 3, collapse_threshold: int = 6):
        """Initialize the generator with context lines and collapse threshold.

        :param context_lines: The number of context lines to show around unchanged lines.
        :type context_lines: int
        :param collapse_threshold: The number of unchanged lines to collapse.
        :type collapse_threshold: int
        """
        self.CONTEXT_LINES = context_lines
        self.COLLAPSE_THRESHOLD = collapse_threshold
        
    def _format_line(self, line: str, prefix: str, old_num: Optional[int] = None, new_num: Optional[int] = None) -> str:
        """Format a single line of the diff with line numbers and prefix.
        
        :param line: The line of code.
        :type line: str
        :param prefix: The prefix indicating the type of change (' ', '-', '+').
        :type prefix: str
        :param old_num: The line number in the old code.
        :type old_num: Optional[int]
        :param new_num: The line number in the new code.
        :type new_num: Optional[int]
        :return: The formatted line.
        :rtype: str
        """
        old_num_str = f"{old_num:4d}" if old_num is not None else "    "
        new_num_str = f"{new_num:4d}" if new_num is not None else "    "
        return f"{old_num_str} {new_num_str} {prefix} {line}"
    
    def _format_collapse(self, skipped_lines: int) -> str:
        """Format a collapse indicator line.
        
        :param skipped_lines: The number of skipped lines.
        :type skipped_lines: int
        :return: The formatted collapse line.
        :rtype: str
        """
        return f"     ⋮    ⋮      ... {skipped_lines} unchanged lines ..."
    
    def generate_diff(self, old_code: str, new_code: str) -> str:
        """Generate a formatted code diff between two pieces of code.
        
        :param old_code: The old code.
        :type old_code: str
        :param new_code: The new code.
        :type new_code: str
        :return: The formatted code diff.
        :rtype: str
        """
        old_lines = old_code.splitlines()
        new_lines = new_code.splitlines()
        matcher = SequenceMatcher(None, old_lines, new_lines)
        
        diff_lines = []
        old_line_num = 1
        new_line_num = 1
        
        for opcode in matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode
            
            if tag == 'equal':
                lines = old_lines[i1:i2]
                if len(lines) > self.COLLAPSE_THRESHOLD:
                    # Add first context lines
                    for line in lines[:self.CONTEXT_LINES]:
                        diff_lines.append(self._format_line(
                            line, " ", old_line_num, new_line_num
                        ))
                        old_line_num += 1
                        new_line_num += 1
                    
                    # Add collapse indicator
                    skipped_lines = len(lines) - (self.CONTEXT_LINES * 2)
                    if skipped_lines > 0:
                        diff_lines.append(self._format_collapse(skipped_lines))
                        old_line_num += skipped_lines
                        new_line_num += skipped_lines
                    
                    # Add last context lines
                    for line in lines[-self.CONTEXT_LINES:]:
                        diff_lines.append(self._format_line(
                            line, " ", old_line_num, new_line_num
                        ))
                        old_line_num += 1
                        new_line_num += 1
                else:
                    for line in lines:
                        diff_lines.append(self._format_line(
                            line, " ", old_line_num, new_line_num
                        ))
                        old_line_num += 1
                        new_line_num += 1
            
            elif tag == 'delete':
                for line in old_lines[i1:i2]:
                    diff_lines.append(self._format_line(
                        line, "-", old_line_num, None
                    ))
                    old_line_num += 1
            
            elif tag == 'insert':
                for line in new_lines[j1:j2]:
                    diff_lines.append(self._format_line(
                        line, "+", None, new_line_num
                    ))
                    new_line_num += 1
            
            elif tag == 'replace':
                # Show as deletion followed by addition
                for line in old_lines[i1:i2]:
                    diff_lines.append(self._format_line(
                        line, "-", old_line_num, None
                    ))
                    old_line_num += 1
                for line in new_lines[j1:j2]:
                    diff_lines.append(self._format_line(
                        line, "+", None, new_line_num
                    ))
                    new_line_num += 1
        
        return "\n".join(diff_lines)