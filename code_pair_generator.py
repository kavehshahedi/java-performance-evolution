from typing import Dict, List, Set, Optional, Any
import json
from git import Repo
import subprocess
import hashlib
import os
from pathlib import Path

class CodePairGenerator:
    """
    A class to generate code pairs from Java method mappings using git repository.
    
    This class handles the process of:
    1. Loading method mappings from a JSON file
    2. Cloning git repository if not exists
    3. Generating unique hashes for method pairs
    4. Simplifying Java signatures
    5. Extracting and saving method implementations as pairs
    """
    
    def __init__(self, project_name: str, git_url: str):
        """
        Initialize the CodePairGenerator.
        
        Args:
            project_name (str): Name of the project directory
            git_url (str): URL of the git repository
        """
        self.project_name = project_name
        self.method_mappings_path = os.path.join('results', project_name, 'method_mappings.json')
        self.git_url = git_url
        self.output_dir = os.path.join('results', project_name, 'code-pairs')
        self.history: Set[str] = set()
        
        # Load method mappings
        self.method_mappings = self._load_method_mappings()
        
        # Setup repository
        self.repo = self._setup_repository()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_method_mappings(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load and return method mappings from JSON file.
        
        :return: Method mappings
        :rtype: Dict[str, List[Dict[str, Any]]]
        """
        try:
            with open(self.method_mappings_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Method mappings file not found: {self.method_mappings_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in method mappings file: {self.method_mappings_path}")

    def _setup_repository(self) -> Repo:
        """Clone the repository if it doesn't exist, or return existing repo.
        
        :return: The git repository object
        :rtype: Repo
        """
        repo_path = os.path.join(os.getcwd(), 'projects', self.project_name)
        if not os.path.exists(repo_path):
            print(f"Cloning repository: {self.git_url}")
            return Repo.clone_from(self.git_url, repo_path)
        return Repo(repo_path)

    @staticmethod
    def generate_unique_hash(input_string: str) -> str:
        """Generate a SHA-256 hash from input string.
        
        :param input_string: The input string to hash
        :type input_string: str
        :return: The SHA-256 hash of the input string
        :rtype: str
        """
        input_bytes = input_string.encode('utf-8')
        hash_object = hashlib.sha256(input_bytes)
        return hash_object.hexdigest()
    
    @staticmethod
    def remove_generic_parameters(signature):
        """
        Remove generic parameters from a method signature.
        
        :param signature: The method signature to remove generics from
        :type signature: str
        :return: The method signature without generic parameters
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

    @staticmethod
    def simplify_java_signature(signature: str) -> str:
        """
        Simplify a Java method signature by removing generics and simplifying class names.
        
        :param signature: The Java method signature to simplify
        :type signature: str
        :return: The simplified Java method signature
        :rtype: str
        """
        # Remove throws clause
        last_paren_index = signature.rfind(')')
        if last_paren_index != -1:
            signature = signature[:last_paren_index + 1]
        
        # Remove generics
        signature = CodePairGenerator.remove_generic_parameters(signature)

        # Split into method and argument parts
        method_parts = signature.split('(')[0].strip().split()
        argument_parts = signature.split('(')[1].split(')')[0].strip().split(',')

        # Simplify method parts
        for idx, mp in enumerate(method_parts):
            if '.' in mp:
                mp = mp.split('.')[-1]
            if '$' in mp:
                mp = mp.split('$')[0]
            method_parts[idx] = mp

        # Simplify argument parts
        simplified_args = []
        for ap in argument_parts:
            if ap.strip() == '':
                continue
            if '.' in ap:
                ap = ap.split('.')[-1]
            if '$' in ap:
                ap = ap.split('$')[0]
            simplified_args.append(ap.split()[0])

        return f"{' '.join(method_parts)}({','.join(simplified_args)})"

    def _extract_method(self, commit_hash: str, file_name: str, method_name: str) -> Optional[str]:
        """
        Extract method implementation from a specific commit and file.
        
        :param commit_hash: The commit hash to checkout
        :type commit_hash: str
        :param file_name: The file name to extract method from
        :type file_name: str
        :param method_name: The method name to extract
        :type method_name: str
        :return: The extracted method implementation
        :rtype: Optional[str]
        """
        if file_name is None or method_name is None:
            return None

        self.repo.git.checkout(commit_hash, force=True)

        process = subprocess.run([
            'java',
            '-jar',
            'jpb.jar',
            '-get-method',
            f'{self.repo.working_dir}/{file_name}',
            self.simplify_java_signature(method_name)
        ], capture_output=True, shell=False)

        if process.returncode != 0:
            print(f'Error: {process.stderr.decode("utf-8")}')
            return None

        output = process.stdout.decode('utf-8').strip()
        if output in ['not-found', 'error']:
            print(f'Error: {output} for {file_name} - {method_name} at {commit_hash}')
            return None

        return output

    def generate_code_pairs(self) -> None:
        """Generate and save code pairs from method mappings."""
        for commit_hash, items in self.method_mappings.items():
            for item in items:
                current_file_name = item['file']
                previous_file_name = item['previous_file']
                current_method_name = item['method_name_cc']
                previous_method_name = item['previous_method_cc']
                current_commit_hash = commit_hash
                previous_commit_hash = item['previous_commit']
                commit_message = item['commit_message']
                significance = item['significance']

                if not significance:
                    continue

                # Generate unique hash for this pair
                hash_ = self.generate_unique_hash(
                    f'{current_commit_hash}-{current_file_name}-{current_method_name}'
                )
                
                # Skip if already processed
                if hash_ in self.history:
                    continue

                # Skip if files already exist
                if all(Path(self.output_dir, f'{hash_}_v{i}.java').exists() for i in [1, 2]):
                    continue

                # Extract method pairs
                method_pairs = []
                for ch, fn, mn in [(current_commit_hash, current_file_name, current_method_name),
                                 (previous_commit_hash, previous_file_name, previous_method_name)]:
                    method_impl = self._extract_method(ch, fn, mn)
                    if method_impl:
                        method_pairs.append(method_impl)

                # Save method pairs if both versions were successfully extracted
                if len(method_pairs) == 2:
                    self.history.add(hash_)
                    for version, implementation in enumerate(method_pairs, 1):
                        output_path = Path(self.output_dir, f'{hash_}_v{version}.java')
                        with open(output_path, 'w') as f:
                            f.write(implementation)

                    # Write metadata to a file
                    metadata = {
                        'hash': hash_,
                        'commit_message': commit_message,
                        'current_commit': current_commit_hash,
                        'previous_commit': previous_commit_hash,
                        'current_file': current_file_name,
                        'previous_file': previous_file_name,
                        'current_method': current_method_name,
                        'previous_method': previous_method_name,
                        'significance': significance
                    }
                    with open(Path(self.output_dir, f'{hash_}_metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=4)