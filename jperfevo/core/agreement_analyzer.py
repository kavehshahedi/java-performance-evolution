import numpy as np
from typing import Dict, List, Set
from sklearn.metrics import cohen_kappa_score

class AgreementAnalyzer:
    def __init__(self, code_labels: List[Dict[str, Set[str]]], reviewers: List[str]):
        """
        Initialize the Aggrement Analyzer with code labels and reviewer names.

        :param code_labels: List of code labels
        :type code_labels: List[Dict[str, Set[str]]]
        :param reviewers: List of reviewer names
        :type reviewers: List[str]
        """
        self.code_labels = code_labels
        self.reviewers = reviewers

    def calculate_kappa(self, code_labels: List[Dict[str, Set[str]]]) -> float:
        """
        Calculate Cohen's Kappa score for a list of code labels.
        
        :param code_labels: List of code labels
        :type code_labels: List[Dict[str, Set[str]]]
        :return: Mean Cohen's Kappa score
        :rtype: float
        """
        # Get all unique categories across all code_labels
        all_categories = sorted(list(set(
            cat for review in code_labels 
            for label_set in self.reviewers
            for cat in label_set
        )))
        
        n_code_labels = len(code_labels)
        n_categories = len(all_categories)
        category_to_idx = {cat: i for i, cat in enumerate(all_categories)}
        
        # Create binary matrices for both reviewers
        reviewer1_matrix = np.zeros((n_code_labels, n_categories))
        reviewer2_matrix = np.zeros((n_code_labels, n_categories))
        
        # Fill matrices with binary values indicating presence/absence of each category
        for i, review in enumerate(code_labels):
            for cat in self.reviewers[0]:
                reviewer1_matrix[i, category_to_idx[cat]] = 1
            for cat in self.reviewers[1]:
                reviewer2_matrix[i, category_to_idx[cat]] = 1
        
        # Calculate kappa for each category and return mean
        category_kappas = []
        for j in range(n_categories):
            try:
                kappa = cohen_kappa_score(reviewer1_matrix[:, j], reviewer2_matrix[:, j])
                if not np.isnan(kappa):
                    category_kappas.append(kappa)
            except:
                continue
        
        return float(np.mean(category_kappas)) if category_kappas else 0.0