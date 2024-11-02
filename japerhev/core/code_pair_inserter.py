import os
import json
import random
import requests
from typing import List

from japerhev.models.code_pair import CodePair


class CodePairInserter:
    """Class to process code pairs and import them to the API.
    It processes all projects in a base directory and extracts all valid code pairs.
    Then, it sends the code pairs to the API endpoint for import.

    :param base_directory: Base directory containing all projects
    :type base_directory: str
    :param api_url: URL of the API endpoint to import code pairs
    :type api_url: str
    """

    def __init__(self, base_directory: str, api_url: str = "http://localhost:5001/api/admin/import-code-pairs"):
        self.base_directory = base_directory
        self.api_url = api_url

    def process_code_pairs_directory(self, code_pairs_path: str, project_name: str) -> List[CodePair]:
        """Process a single code-pairs directory and extract all valid code pairs.
        
        :param code_pairs_path: Path to the code-pairs directory
        :type code_pairs_path: str
        :param project_name: Name of the project
        :type project_name: str
        :return: List of CodePair objects
        :rtype: List[CodePair]
        """
        code_pairs = []
        files = os.listdir(code_pairs_path)

        for file in files:
            if not file.endswith('_v1.java'):
                continue
                
            base_name = file.replace('_v1.java', '')
            v2_file = f"{base_name}_v2.java"
            metadata_file = f"{base_name}_metadata.json"

            if v2_file not in files or metadata_file not in files:
                continue

            try:
                # Read all files
                with open(os.path.join(code_pairs_path, file), 'r', encoding='utf-8') as f:
                    v1_content = f.read()
                with open(os.path.join(code_pairs_path, v2_file), 'r', encoding='utf-8') as f:
                    v2_content = f.read()
                with open(os.path.join(code_pairs_path, metadata_file), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                commit_hash = metadata.get('current_commit')
                if not commit_hash:
                    print(f"Warning: No commit_hash found in metadata for {project_name}/{base_name}")
                    continue

                code_pairs.append(CodePair(
                    projectName=project_name,
                    version1=v1_content,
                    version2=v2_content,
                    commitHash=commit_hash,
                    commitMessage=metadata.get('commit_message', ''),
                    performanceChange=metadata.get('significance', {}).get('change_type', '')
                ))

            except Exception as e:
                print(f"Warning: Error processing {project_name}/{base_name}: {str(e)}")

        return code_pairs

    def import_code_pairs(self):
        """Process all projects and import code pairs to the API."""
        if not os.path.exists(self.base_directory):
            raise FileNotFoundError(f"Base directory not found: {self.base_directory}")

        # Get all project directories
        project_dirs = [d for d in os.listdir(self.base_directory) 
                       if os.path.isdir(os.path.join(self.base_directory, d))]
        
        all_code_pairs = []

        # Process each project
        for project_dir in project_dirs:
            project_path = os.path.join(self.base_directory, project_dir)
            code_pairs_path = os.path.join(project_path, 'code-pairs')

            if not os.path.exists(code_pairs_path):
                print(f"Warning: No valid code-pairs directory found for project: {project_dir}")
                continue

            print(f"Processing project: {project_dir}")
            project_code_pairs = self.process_code_pairs_directory(code_pairs_path, project_dir)
            all_code_pairs.extend(project_code_pairs)

        # Apply a resampling strategy
        significant_changes = []
        insignificant_changes = []
        for cp in all_code_pairs:
            if cp.performanceChange != "unchanged":
                significant_changes.append(cp)
            else:
                insignificant_changes.append(cp)

        n_samples = len(significant_changes)
        insignificant_changes = random.sample(insignificant_changes, n_samples)
        all_code_pairs = significant_changes + insignificant_changes
        
        # Shuffle the list
        random.shuffle(all_code_pairs)

        if not all_code_pairs:
            raise ValueError('No valid code pairs found in any project')

        # Convert CodePair objects to dictionaries for API request
        code_pairs_data = [vars(cp) for cp in all_code_pairs]

        try:
            # Send code pairs to the API
            response = requests.post(self.api_url, json={'codePairs': code_pairs_data})
            response.raise_for_status()
            
            print(response.json().get('message', 'Success'))
            print(f"Successfully processed {len(all_code_pairs)} code pairs from {len(project_dirs)} projects")
            
        except Exception as e:
            print(f"Error importing code pairs: {str(e)}")