from abc import abstractmethod, ABC
from typing import Optional
from pydantic import BaseModel

class VectorEntry(BaseModel):
    id: Optional[str] = None
    content: str


class VecStore(ABC):
    """
    Abstract class for Vector Stores.
    """

    @abstractmethod
    def create_vectorstore(self):
        """
        Create the vector store instance.
        """
        pass

    @abstractmethod
    def get_vectorstore(self, create=True):
        """
        Get the vector store instance.
        """
        pass

    @abstractmethod
    def add_items(self, items: list) -> None:
        """
        Add items to the vector store.

        Args:
            items (list): List of items to add.

        """
        pass

    @abstractmethod
    def delete_items(self, ids: list) -> None:
        """
        Delete items from a vector store given ids.

        Args:
            src_id (str): The source ID corresponding to each item.
            user_id(str): The user ID that the item belong to.
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """
        Delete the vector store.
        """
        pass
