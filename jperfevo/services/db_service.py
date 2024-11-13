from typing import Dict, Optional, List
from bson import ObjectId
from pymongo import MongoClient
import os
import dotenv

from jperfevo.models.code_pair import CodePair

dotenv.load_dotenv(override=True)


class DBService:
    """
    This module is responsible for handling the database operations.
    """

    def __init__(self, db_name: str = os.getenv('DB_NAME', 'cctb'),
                    db_url: str = os.getenv('DB_URL', 'localhost:27017'),
                    use_cloud_db: bool = True) -> None:
        """
        Initialize the database service.

        :param db_name: Name of the database
        :type db_name: str
        :param db_url: URL of the database
        :type db_url: str
        :param use_cloud_db: Flag to use cloud database
        :type use_cloud_db: bool
        """
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
        cursor = self.db['codepairs'].find(query, {'__v': 0})

        for pair in cursor:
            code_pairs.append(CodePair(**pair))

        return code_pairs
    
    def get_reviews(self, user_id: Optional[str] = None, review_id: Optional[str] = None) -> List[Dict]:
        """Get reviews from the database.
        
        :param user_id: ID of the user to filter by
        :type user_id: str
        :param review_id: ID of the review to filter by
        :type review_id: str
        :return: List of reviews
        :rtype: List[Dict]
        """
        query = {}
        if user_id:
            query['userId'] = ObjectId(user_id)
        if review_id:
            query['reviewId'] = ObjectId(review_id)

        cursor = self.db['codereviews'].find(query)

        return list(cursor)