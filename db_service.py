from typing import Optional, List
from pymongo import MongoClient
import os
import dotenv

from .code_pair import CodePair

dotenv.load_dotenv()


class DBService:
    """
    This module is responsible for handling the database operations.
    """

    def __init__(self, db_name: str = os.getenv('DB_NAME', 'cctb'),
                    db_url: str = os.getenv('DB_URL', 'localhost:27017'),
                    use_cloud_db: bool = False) -> None:
        if use_cloud_db:
            db_url = os.getenv('CLOUD_DB_URL', db_url)
 
        self.client = MongoClient(db_url)
        self.db = self.client[db_name]

    def get_code_pairs(self, project_name: Optional[str] = None) -> List[CodePair]:
        """Get all code pairs from the database.
        
        :param project_name: Name of the project to filter by
        :type project_name: str
        :return: List of CodePair objects
        :rtype: List[CodePair]
        """
        code_pairs = []
        query = {} if project_name is None else {'project_name': project_name}
        cursor = self.db.code_pairs.find(query)

        for pair in cursor:
            code_pairs.append(CodePair(**pair))

        return code_pairs