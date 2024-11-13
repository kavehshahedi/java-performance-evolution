from typing import Dict, Optional
import requests
from datetime import datetime
import math
import time

class GitHubAuthorExperience:
    """
    A class to calculate the experience of a GitHub author based on various metrics.

    We consider the following metrics:
    - Total contributions
    - Repository contributions
    - Code reviews
    - Account age
    """

    def __init__(self, token: str) -> None:
        """
        Initialize the GitHubAuthorExperience class with a GitHub token.

        :param token: GitHub token for authentication
        :type token: str
        """
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json"
        }

        self.weights = {
            "repo_score": 0.3,
            "total_contributions": 0.25,
            "reviews": 0.25,
            "account_age": 0.2
        }

        self.project_contributions = {}

    def get_author_experience(self, repo: str, commit_sha: str, defined_author_username: Optional[str] = None) -> Optional[Dict]:
        """
        Get the experience assessment of a GitHub author based on the provided metrics.

        :param repo: Repository name in the format 'owner/repo'
        :type repo: str
        :param commit_sha: Commit SHA for the author
        :type commit_sha: str
        :param defined_author_username: Optional defined author username (if available)
        :type defined_author_username: Optional[str]
        :return: Structured experience assessment
        :rtype: Optional[Dict]
        """
        # Get the author's username from the commit
        if not defined_author_username:
            author_username = self.get_commit_author(repo, commit_sha)
            if not author_username:
                return
        else:
            author_username = defined_author_username

        try:
            user_details = self._get_user_details(author_username)
            repo_contributions = self._get_repo_contributions(author_username, repo)
            total_contributions = self._get_contributions(author_username)
            code_reviews = self._get_total_code_reviews(author_username)

            experience_score = self._calculate_experience_score(user_details, repo_contributions, total_contributions, code_reviews)

            # Return structured experience assessment
            return {
                "username": author_username,
                "total_contributions": total_contributions,
                "repo_contributions": repo_contributions,
                "code_reviews": code_reviews,
                "account_age_years": (datetime.now() - datetime.strptime(user_details['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days / 365,
                "experience_score": experience_score
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {author_username}: {str(e)}")
            return None

    def get_commit_author(self, repo: str, commit_sha: str) -> Optional[str]:
        """
        Get the author's GitHub username from a commit SHA.
        
        :param repo: Repository name in the format 'owner/repo'
        :type repo: str
        :param commit_sha: Commit SHA for the author
        :type commit_sha: str
        :return: Author's GitHub username
        :rtype: Optional[str]
        """
        url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        commit_data = response.json()

        # Get author's GitHub username (if available)
        author = commit_data.get('author')
        if author and 'login' in author:
            return author['login']
        else:
            # Fallback to committer's username
            committer = commit_data.get('committer')
            if committer and 'login' in committer:
                return committer['login']
            
        return None

    def _get_user_details(self, username: str) -> Dict:
        """
        Get the details of a GitHub user based on their username.

        :param username: GitHub username
        :type username: str
        :return: User details
        :rtype: Dict
        """
        url = f"https://api.github.com/users/{username}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def _get_contributions(self, username: str) -> int:
        """
        Get the total contributions of a GitHub user.

        :param username: GitHub username
        :type username: str
        :return: Total contributions
        :rtype: int
        """
        url = f"https://github-contributions-api.deno.dev/{username}.json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['totalContributions']

    def _get_repo_contributions(self, username: str, repository: str) -> int:
        """
        Get the contributions of a GitHub user to a specific repository.

        :param username: GitHub username
        :type username: str
        :param repository: Repository name in the format 'owner/repo'
        :type repository: str
        :return: Contributions to the repository
        :rtype: int
        """
        owner, repo = str(repository).split('/')
        
        if repository in self.project_contributions and self.project_contributions[repository]:
            contributor = next((c for c in self.project_contributions[repository] if c['author']['login'] == username), None)
            if contributor:
                return contributor['total']
            else:
                return 0

        url = f"https://api.github.com/repos/{owner}/{repo}/stats/contributors"

        # Retry until the API is fully loaded
        max_retries = 10
        for attempt in range(max_retries):
            response = requests.get(url, headers=self.headers)
            if response.status_code == 202:  # GitHub returns 202 if the stats are being generated
                print("Stats are being generated, retrying...")
                time.sleep(2 ** attempt)  
                continue
            response.raise_for_status()
            contributors = response.json()
            break
        else:
            return 0

        try:
            # Find the user's contributions
            contributor = next((c for c in contributors if c['author']['login'] == username), None)
            self.project_contributions[repository] = contributors
            if contributor:
                return contributor['total']
        except:
            return 0
        
        return 0
    
    def _get_total_code_reviews(self, username: str) -> int:
        """
        Get the total number of code reviews done by a GitHub user.

        :param username: GitHub username
        :type username: str
        :return: Total code reviews
        :rtype: int
        """
        url = f"https://api.github.com/search/issues?q=type:pr+author:{username}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        if 'total_count' in data:
            return data['total_count']
        return 0

    def _calculate_experience_score(self, user_details: Dict,
                                      repo_contributions: int,
                                      total_contributions: int,
                                      code_reviews: int) -> Optional[float]:
        """
        Calculate the experience score of a GitHub author based on the provided metrics.

        :param user_details: Details of the GitHub user
        :type user_details: Dict
        :param repo_contributions: Contributions to the repository
        :type repo_contributions: int
        :param total_contributions: Total contributions of the user
        :type total_contributions: int
        :param code_reviews: Total code reviews done by the user
        :type code_reviews: int
        :return: Experience score
        :rtype: float
        """
        # Extract weights
        repo_weight = self.weights.get("repo_score", 0)
        total_contrib_weight = self.weights.get("total_contributions", 0)
        account_age_weight = self.weights.get("account_age", 0)
        reviews_weight = self.weights.get("reviews", 0)

        # Calculate repo score with diminishing returns
        repo_score = 1 - math.exp(-repo_contributions / 50)  # Smoother scaling up to 100+

        # Calculate total contributions score with logarithmic scaling
        contrib_score = min(math.log10(total_contributions + 1) / 2, 1.0)

        # Calculate code reviews score with diminishing returns
        reviews_score = 1 - math.exp(-code_reviews / 50)  # Plateaus around 100+ reviews

        # Calculate account age score with diminishing returns
        days_since_creation = (datetime.now() - 
                            datetime.strptime(user_details['created_at'], 
                                            '%Y-%m-%dT%H:%M:%SZ')).days
        years = days_since_creation / 365
        age_score = 1 - math.exp(-years / 3)  # Smoother scaling, plateaus around 10 years

        # Consider recent activity
        if 'updated_at' in user_details:
            days_since_last_activity = (datetime.now() - 
                                    datetime.strptime(user_details['updated_at'], 
                                                    '%Y-%m-%dT%H:%M:%SZ')).days
            activity_factor = math.exp(-days_since_last_activity / 365)  # Decay over a year
        else:
            activity_factor = 1.0

        # Calculate final score including code reviews
        score = (repo_weight * repo_score +
                total_contrib_weight * contrib_score +
                account_age_weight * age_score +
                reviews_weight * reviews_score) * activity_factor

        return min(score, 1.0)