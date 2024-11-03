from dataclasses import dataclass
from typing import Optional

@dataclass
class CodePair:
    """Data class to represent a code pair.
    
    :param projectName: Name of the project
    :type projectName: str
    :param version1: Content of the first version
    :type version1: str
    :param version2: Content of the second version
    :type version2: str
    :param commitHash: Commit hash of the code pair
    :type commitHash: str
    :param commitMessage: Commit message of the code pair
    :type commitMessage: str
    :param performanceChange: Performance change of the code pair
    :type performanceChange: str
    """
    projectName: str
    version1: str
    version2: str
    commitHash: str
    commitMessage: str
    performanceChange: str
    _id: Optional[str] = None