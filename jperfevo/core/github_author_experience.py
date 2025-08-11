from typing import Dict, List, Optional
import requests
from datetime import datetime
import math
import time


class GitHubRateLimitError(Exception):
    pass


class GitHubAuthorExperience:
    def __init__(self, token: str, rate_handler=None) -> None:
        self.token = token
        self.rate_handler = rate_handler
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

    def _make_request(self, url: str, custom_headers: Optional[Dict] = None) -> requests.Response:
        """Make a request with rate limit handling and token rotation on any error."""
        headers = custom_headers or self.headers
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)

                if response.status_code in [403, 429]:
                    if self.rate_handler:
                        self.token = self.rate_handler.rotate_token_on_error()
                        self.headers["Authorization"] = f"Bearer {self.token}"
                        headers = custom_headers or self.headers
                        continue
                    else:
                        raise GitHubRateLimitError(f"Rate limit hit: {response.text}")

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if self.rate_handler:
                    self.token = self.rate_handler.rotate_token_on_error()
                    self.headers["Authorization"] = f"Bearer {self.token}"
                    headers = custom_headers or self.headers
                if attempt == max_retries - 1:
                    raise GitHubRateLimitError(f"Request failed after {max_retries} attempts: {str(e)}")
                time.sleep(2 ** attempt)

        raise GitHubRateLimitError("Max retries exceeded")

    def get_author_experience(self, repo: str, commit_sha: str, defined_author_username: Optional[str] = None) -> Optional[Dict]:
        try:
            commit_info = self._get_commit_info(repo, commit_sha)
            if not commit_info:
                return None

            commit_date = commit_info['commit_date']
            author_username = defined_author_username or commit_info['username']
            if not author_username:
                return None

            user_details = self._get_user_details(author_username)
            repo_contributions = abs(self._get_repo_contributions(author_username, repo, commit_date))
            total_contributions = abs(self._get_total_contributions(author_username, commit_date))
            code_reviews = abs(self._get_total_code_reviews(author_username, commit_date))

            created_at = datetime.strptime(user_details['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            account_age_years = abs((commit_date - created_at).days / 365)

            experience_score = self._calculate_experience_score(
                repo_contributions=repo_contributions,
                total_contributions=total_contributions,
                code_reviews=code_reviews,
                account_age_years=account_age_years
            )

            return {
                "username": author_username,
                "total_contributions": total_contributions,
                "repo_contributions": repo_contributions,
                "code_reviews": code_reviews,
                "account_age_years": account_age_years,
                "experience_score": experience_score
            }

        except Exception as e:
            print(f"Error fetching data for commit {commit_sha}: {str(e)}")
            return None

    def _get_commit_info(self, repo: str, commit_sha: str) -> Optional[Dict]:
        url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
        response = self._make_request(url)
        commit_data = response.json()

        author = commit_data.get('author', {})
        committer = commit_data.get('committer', {})
        username = author.get('login') if author else committer.get('login') if committer else None

        if not username:
            author = commit_data.get('commit', {}).get('author', {})
            commiter = commit_data.get('commit', {}).get('committer', {})
            username = author.get('name') if author else commiter.get('name') if commiter else None

        if not username:
            return None

        # Case-specific handling for Zipkin project
        if username == 'Adrian Cole' and repo.lower() == 'openzipkin/zipkin':
            username = 'adriancole'

        commit_date_str = commit_data['commit']['author']['date']
        commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
        return {'username': username, 'commit_date': commit_date}

    def _get_user_details(self, username: str) -> Dict:
        url = f"https://api.github.com/users/{username}"
        response = self._make_request(url)
        return response.json()

    def _get_total_contributions(self, username: str, commit_date: datetime) -> int:
        commit_date_str = commit_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = f"https://api.github.com/search/commits?q=author:{username}+author-date:<={commit_date_str}"
        headers = {**self.headers, "Accept": "application/vnd.github.cloak-preview"}
        response = self._make_request(url, headers)
        return response.json().get('total_count', 0)

    def _get_repo_contributions(self, username: str, repo: str, commit_date: datetime) -> int:
        owner, repo_name = repo.split('/')
        cache_key = (repo, commit_date.timestamp())
        if cache_key in self.project_contributions:
            contributors = self.project_contributions[cache_key]
        else:
            contributors = self._fetch_contributors_with_retry(owner, repo_name)
            self.project_contributions[cache_key] = contributors

        commit_timestamp = commit_date.timestamp()
        for contributor in contributors:
            if contributor['author']['login'] == username:
                return sum(week['c'] for week in contributor['weeks'] if week['w'] <= commit_timestamp)
        return 0

    def _fetch_contributors_with_retry(self, owner: str, repo: str, max_retries: int = 10) -> list:
        url = f"https://api.github.com/repos/{owner}/{repo}/stats/contributors"

        for attempt in range(max_retries):
            try:
                response = self._make_request(url)
                if response.status_code == 202:
                    time.sleep(2 ** attempt)
                    continue
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise GitHubRateLimitError(f"Failed to fetch contributors after {max_retries} retries: {str(e)}")
                time.sleep(2 ** attempt)
        raise GitHubRateLimitError("Max retries exceeded for fetching contributors")

    def _get_total_code_reviews(self, username: str, commit_date: datetime) -> int:
        commit_date_str = commit_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        url = f"https://api.github.com/search/issues?q=type:pr+author:{username}+created:<={commit_date_str}"
        response = self._make_request(url)
        return response.json().get('total_count', 0)

    def _calculate_experience_score(self, repo_contributions: int, total_contributions: int,
                                    code_reviews: int, account_age_years: float) -> float:
        repo_score = 1 - math.exp(-repo_contributions / 50)
        contrib_score = min(math.log10(total_contributions + 1) / 2, 1.0)
        reviews_score = 1 - math.exp(-code_reviews / 50)
        age_score = 1 - math.exp(-account_age_years / 3)

        weighted_score = (
            self.weights["repo_score"] * repo_score +
            self.weights["total_contributions"] * contrib_score +
            self.weights["reviews"] * reviews_score +
            self.weights["account_age"] * age_score
        )
        return min(weighted_score, 1.0)


class GitHubRateLimitHandler:
    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.current_token_idx = 0
        self.token_errors = {token: 0 for token in tokens}
        self.max_errors = 3

    def get_current_token(self) -> str:
        return self.tokens[self.current_token_idx]

    def rotate_token(self) -> str:
        self.current_token_idx = (self.current_token_idx + 1) % len(self.tokens)
        return self.get_current_token()

    def rotate_token_on_error(self) -> str:
        current_token = self.get_current_token()
        self.token_errors[current_token] += 1

        if self.token_errors[current_token] >= self.max_errors:
            print(f"Token {self.current_token_idx + 1} hit max errors, rotating...")
            new_token = self.rotate_token()
            self.token_errors[current_token] = 0
            return new_token

        return current_token

    def check_rate_limit(self, token: str) -> Dict:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
        try:
            response = requests.get("https://api.github.com/rate_limit", headers=headers)
            response.raise_for_status()
            return response.json()['resources']['core']
        except requests.exceptions.RequestException:
            return {'remaining': 0, 'reset': time.time() + 3600}

    def wait_for_reset_if_needed(self) -> str:
        while True:
            current_token = self.get_current_token()
            rate_info = self.check_rate_limit(current_token)

            remaining = rate_info['remaining']
            if remaining > 100:
                return current_token

            if all(self.check_rate_limit(token)['remaining'] <= 100 for token in self.tokens):
                earliest_reset = min(
                    self.check_rate_limit(token)['reset']
                    for token in self.tokens
                )
                wait_time = earliest_reset - time.time() + 5
                if wait_time > 0:
                    print(f"All tokens near limit. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)

            self.rotate_token()
