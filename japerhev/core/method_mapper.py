import json
import os
import sys
import re
from typing import Dict, List, Tuple, Any, Optional

from japerhev.services.similarity_service import SimilarityService

MINIMUM_CALL_COUNT = 15
MIN_PERFORMANCE_DIFF = 0.05


class MethodMapper:
    def __init__(self, candidate_commits_file: str, performance_data_file: str):
        """
        Initialize the MethodMapper with file paths for candidate commits and performance data.

        :param candidate_commits_file: Path to the JSON file containing candidate commits
        :type candidate_commits_file: str
        :param performance_data_file: Path to the JSON file containing performance data
        :type performance_data_file: str
        """
        self.candidate_commits: Dict[str, Any] = self._convert_list_to_dict(self._load_json(candidate_commits_file))
        self.performance_data: Dict[str, Any] = self._load_json(performance_data_file)

        self.converted_method_history: Dict[str, str] = {}
        self.tokenized_method_history: Dict[str, List[str]] = {}


    @staticmethod
    def _load_json(filename: str) -> Any:
        """
        Load JSON data from a file.

        :param filename: Path to the JSON file
        :type filename: str
        :return: Loaded JSON data
        :rtype: Any
        :raises JSONDecodeError: If the file contains invalid JSON
        :raises FileNotFoundError: If the file is not found
        """
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {filename}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        
    @staticmethod
    def _convert_list_to_dict(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert a list of dictionaries to a dictionary using a specified key.

        :param data: List of dictionaries
        :type data: List[Dict[str, Any]]
        :param key: Key to use as the dictionary key
        :type key: str
        :return: Dictionary with the specified key
        :rtype: Dict[str, Any]
        """
        return {commit['commit']: commit for commit in data}

    @staticmethod
    def _save_json(data: Any, filename: str) -> None:
        """
        Save data to a JSON file.

        :param data: Data to be saved
        :type data: Any
        :param filename: Path to the output JSON file
        :type filename: str
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def tokenize_method_signature(self, signature: str) -> List[str]:
        """
        Tokenize a method signature into words, considering camelCase and snake_case.

        :param signature: Method signature to tokenize
        :type signature: str
        :return: List of lowercase tokens
        :rtype: List[str]
        """
        if signature in self.tokenized_method_history:
            return self.tokenized_method_history[signature]
        
        signature = signature.strip()
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', signature)
        tokens = [token.lower() for token in tokens]

        self.tokenized_method_history[signature] = tokens

        return tokens
    
    @staticmethod
    def remove_generic_parameters(signature) -> str:
        """
        Remove generic type parameters from a method signature.

        :param signature: Method signature to process
        :type signature: str
        :return: Method signature without generic type parameters
        :rtype: str
        """
        result = []
        bracket_count = 0
        skip = False
        
        for char in signature:
            if char == '<':
                bracket_count += 1
                skip = True
            elif char == '>':
                bracket_count -= 1
                if bracket_count == 0:
                    skip = False
                continue
                
            if not skip:
                result.append(char)
                
        return ''.join(result)

    def convert_method_signature(self, method_signature: str) -> str:
        """
        Convert a method signature to a standardized format.

        :param method_signature: Original method signature
        :type method_signature: str
        :return: Converted method signature
        :rtype: str
        """
        if method_signature in self.converted_method_history:
            return self.converted_method_history[method_signature]

        # Remove throws clause
        last_paren_index = method_signature.rfind(')')
        if last_paren_index != -1:
            method_signature = method_signature[:last_paren_index + 1]
        
        # Remove generics
        method_signature = self.remove_generic_parameters(method_signature)

        # Split into method and argument parts
        if '(' not in method_signature:
            return method_signature.lower().replace(' ', '').strip()
        
        method_parts, argument_parts = method_signature.split('(', 1)
        method_parts = method_parts.strip().split()
        argument_parts = argument_parts.split(')')[0].strip()

        # Process method parts
        method_parts = [part.split('.')[-1].split('$')[-1] for part in method_parts]

        # Process arguments
        arguments = [arg.strip() for arg in argument_parts.split(',') if arg.strip()]
        arguments = [arg.split('.')[-1].split('$')[-1].split()[0] for arg in arguments]

        # Reconstruct the method signature
        method_parts = ' '.join(method_parts)
        argument_parts = ','.join(arguments)
        entire = f"{method_parts}({argument_parts})"

        converted = entire.lower().replace(' ', '').strip()

        self.converted_method_history[method_signature] = converted

        return converted

    def find_previous_method(self, current_commit: str, current_method: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the previous version of a method in the commit history.

        :param current_commit: Current commit hash
        :type current_commit: str
        :param current_method: Current method name
        :type current_method: str
        :return: Tuple of (previous method name, previous file name) or (None, None) if not found
        :rtype: Tuple[Optional[str], Optional[str]]
        """
        current_tokens = self.tokenize_method_signature(current_method)
        best_similarity = 0
        best_method_name = None
        best_file_name = None

        commit = self.candidate_commits[current_commit]
        previous_commit = commit['previous_commit']
        previous_method_changes = commit['method_changes'].get(previous_commit, {})

        for file_name, method_names in previous_method_changes.items():
            for method_name in method_names:
                if current_method == method_name:
                    return method_name, file_name
                else:
                    previous_tokens = self.tokenize_method_signature(method_name)
                    ss = SimilarityService(current_tokens, previous_tokens)
                    are_similar, similarity_score = ss.are_similar()
                    if are_similar and similarity_score > best_similarity:
                        best_similarity = similarity_score
                        best_method_name = method_name
                        best_file_name = file_name

        return best_method_name, best_file_name

    def create_method_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create method mappings between performance data and candidate commits.

        :return: Dictionary of method mappings
        :rtype: Dict[str, List[Dict[str, Any]]]
        """
        method_mappings: Dict[str, List[Dict[str, Any]]] = {}

        for pd_commit, pd_data in self.performance_data.items():
            for pd_commit_hash, pd_benchmark in pd_data.items():
                for pd_benchmark_name, pd_called_methods in pd_benchmark.items():
                    for pd_method_name in pd_called_methods.keys():
                        pd_method_name_converted = self.convert_method_signature(pd_method_name)
                        found_mapping = self._find_mapping_for_method(pd_commit, pd_method_name, pd_method_name_converted, pd_benchmark_name, pd_called_methods[pd_method_name])
                        if found_mapping:
                            if pd_commit_hash not in method_mappings:
                                method_mappings[pd_commit_hash] = []
                            method_mappings[pd_commit_hash].append(found_mapping)
                break

        # Remove commits with no method mappings
        method_mappings = {commit: mappings for commit, mappings in method_mappings.items() if mappings}

        return method_mappings

    def _find_mapping_for_method(self, pd_commit: str, pd_method_name:str, pd_method_name_converted: str, pd_benchmark_name: str, current_pd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the mapping for a given method.

        :param pd_commit: Commit hash of the performance data
        :type pd_commit: str
        :param pd_method_name_converted: Converted method name
        :type pd_method_name_converted: str
        :param pd_benchmark_name: Benchmark name
        :type pd_benchmark_name: str
        :param current_pd: Performance data for the current method
        :type current_pd: Dict[str, Any]
        :return: Dictionary containing the mapping, or None if no mapping found
        :rtype: Dict[str, Any]
        """
        cc_commit = self.candidate_commits.get(pd_commit, None)
        if cc_commit is None:
            return {}

        cc_method_changes = cc_commit['method_changes'].get(cc_commit['commit'], {})

        for cc_file_name, cc_method_names in cc_method_changes.items():
            for cc_method_name in cc_method_names:
                cc_method_name_converted = self.convert_method_signature(cc_method_name)
                if pd_method_name_converted == cc_method_name_converted:
                    previous_method, previous_file = self.find_previous_method(cc_commit['commit'], cc_method_name)
                    if previous_method is None or previous_file is None:
                        return {}

                    is_in_pd, pd_method_name_prev, prev_pd = self._is_method_in_performance_data(pd_commit, previous_method)
                    if not is_in_pd or current_pd['call_count'] < MINIMUM_CALL_COUNT or prev_pd['call_count'] < MINIMUM_CALL_COUNT:
                        return {}

                    performance_diff = self.calculate_combined_performance(current_pd['average_self_time'], prev_pd['average_self_time'], current_pd['min_execution_time'], prev_pd['min_execution_time'])
                    # if abs(performance_diff) < MIN_PERFORMANCE_DIFF:
                    #     return {}

                    return {
                        'commit_message': cc_commit['commit_message'],
                        'benchmark': pd_benchmark_name,
                        'method_name_pd': pd_method_name,
                        'method_name_cc': cc_method_name,
                        'file': cc_file_name,
                        'previous_method_cc': previous_method,
                        'previous_method_pd': pd_method_name_prev,
                        'previous_file': previous_file,
                        'previous_commit': cc_commit['previous_commit'],
                        'performance_diff': performance_diff
                    }
        return {}

    def calculate_combined_performance(self, current_avg: float, prev_avg: float, 
                                 current_min: float, prev_min: float, 
                                 avg_weight: float = 0.8, min_weight: float = 0.2) -> float:
        """
        Calculate the combined performance difference based on average and minimum execution times.
        
        :param current_avg: Current average execution time
        :type current_avg: float
        :param prev_avg: Previous average execution time
        :type prev_avg: float
        :param current_min: Current minimum execution time
        :type current_min: float
        :param prev_min: Previous minimum execution time
        :type prev_min: float
        :param avg_weight: Weight for the average execution time
        :type avg_weight: float
        :param min_weight: Weight for the minimum execution time
        :type min_weight: float
        :return: Combined performance difference
        :rtype: float
        """
        avg_ratio = current_avg / prev_avg
        min_ratio = current_min / prev_min

        # Handle negative ratios by taking absolute values and preserving sign
        avg_sign = 1 if avg_ratio >= 0 else -1
        min_sign = 1 if min_ratio >= 0 else -1
        
        # Use absolute values for the geometric mean calculation
        combined_diff = -((abs(avg_ratio) ** avg_weight) * (abs(min_ratio) ** min_weight) - 1)
        
        # Restore the sign based on the weighted combination of input signs
        final_sign = avg_sign * avg_weight + min_sign * min_weight
        return combined_diff * (1 if final_sign >= 0 else -1)

    def _is_method_in_performance_data(self, pd_commit: str, method: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if a method exists in the performance data.

        :param pd_commit: Performance data commit
        :type pd_commit: str
        :param method: Method to search for
        :type method: str
        :return: True if the method is found, False otherwise
        :rtype: bool
        """
        if pd_commit not in self.performance_data:
            return False, '', {}

        converted_method = self.convert_method_signature(method)
        for pd_commit_hash, pd_benchmark in self.performance_data[pd_commit].items():
            for benchmark_name, benchmark_data in pd_benchmark.items():
                for method_name, method_data in benchmark_data.items():
                    if self.convert_method_signature(method_name) == converted_method:
                        return True, method_name, method_data
        return False, '', {}

    def run(self, output_file: str) -> None:
        """
        Run the method mapping process and save the results to a file.

        :param output_file: Path to the output JSON file
        :type output_file: str
        """
        method_mappings = self.create_method_mappings()
        self._save_json(method_mappings, output_file)

def main():
    with open('projects.json', 'r') as file:
        projects = json.load(file)

    for project in projects:
        project_name = project['name']
        print(f"Processing project: {project_name}")
        candidate_commits_path = os.path.join(sys.path[0], 'jphb-performance-data', project_name, 'candidate_commits.json')
        performance_data_path = os.path.join(sys.path[0], 'jphb-performance-data', project_name, 'performance_data.json')
        mapper = MethodMapper(candidate_commits_path, performance_data_path)
        mapper.run(os.path.join('results', project_name, 'method_mappings.json'))


if __name__ == "__main__":
    main()
