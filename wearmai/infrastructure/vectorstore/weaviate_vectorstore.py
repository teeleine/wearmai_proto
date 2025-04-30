from .base import VecStore, VectorEntry
import weaviate
from typing import List
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from weaviate.classes.query import HybridFusion
from wearmai.settings import WEAVIATE_URL, WEAVIATE_API_KEY, VOYAGEAI_API_KEY
import structlog

log = structlog.get_logger(__name__)

class WeaviateVecStore(VecStore):
    def __init__(self, vs_name: str) -> None:
        self.vs_name = vs_name
        self.client = self._connect_remote()
        self.vectorstore = self.get_vectorstore()
    
    def _connect_remote(self) -> weaviate.client.WeaviateClient:
        wclient = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,                                    
            auth_credentials=wvc.init.Auth.api_key(WEAVIATE_API_KEY),    
            headers={"X-VoyageAI-Api-Key": VOYAGEAI_API_KEY} 
        )

        return wclient
    
    def close(self):
        log.info("closed_vectorstore_connection", vs_name=self.vs_name)
        self.client.close()
    
    def _format_search_results(self, rs: List) -> VectorEntry:
        res_lst = []
        for obj in rs.objects:
            properties = obj.properties
            entry = VectorEntry(
                id=str(obj.uuid),
                content=str(properties['content'])
            )
            res_lst.append(entry)
        
        return res_lst
    
    def get_vectorstore(self):
        """
        Get the Weaviate vector store instance.
        """
        existing_collections = self.client.collections.list_all()
        if self.vs_name in existing_collections:
            return self.client.collections.get(self.vs_name)
        new_vec_store = self.create_vectorstore()
        return new_vec_store

    def delete_collection(self) -> None:
        self.client.collections.delete(self.vs_name)
        log.info("vectorstore_collection_deleted", vs_name=self.vs_name) 

    def create_vectorstore(self):
        log.info("creating_vectorstore_collection", vs_name=self.vs_name)
        vectorizer_config = wvc.config.Configure.NamedVectors.text2vec_voyageai(
            name="content_vector",
            source_properties=["content"],
            model="voyage-3"
        )
        properties = [
            wvc.config.Property(
                name="content",
                data_type=wvc.config.DataType.TEXT
            ),
        ]
        new_vec_store = self.client.collections.create(
            name=self.vs_name,
            vectorizer_config=[vectorizer_config],  
            properties=properties,
        )
        log.info("vectorstore_collection_created", vs_name=self.vs_name)
        return new_vec_store

    def format_chunks(self, chunks: list) -> list[dict]: 
        
        chunk_objects = list()

        for i in range(len(chunks)):
            properties = {"content": chunks[i].text}
            data_object = wvc.data.DataObject(
                properties=properties,
                uuid=generate_uuid5(properties)
            )
            chunk_objects.append(data_object)
        
        return chunk_objects
    
    def add_items(self, chunks: List):

        chunk_objects = self.format_chunks(chunks)
        insertion_response = self.vectorstore.data.insert_many(chunk_objects)

        log.info("added_items_to_vectorstore", chunk_count=len(chunk_objects), insertion_response=insertion_response)

    def get(
        self,
        item_id: str = None,
        limit: int = 1000,
        include_vector=False,
    ) -> List[VectorEntry]:
        """
        Get items from a vector store given an id

        Args:
            ids (list): An item ID corresponding to an item that is desired to be deleted.
            limit (int): the number of results to limit the response by
            include_vector (bool): fetch vectors as well
        """
        if item_id:
            rs = self.vectorstore.query.fetch_object_by_id(
                item_id, include_vector=include_vector
            )
        else:
            rs = self.vectorstore.query.fetch_objects(
                limit=limit, include_vector=include_vector
            )

        return self._format_search_results(rs)

    def delete_items(self, ids: list = None) -> None:
        """
        Delete items from a vector store given ids

        Args:
            ids (list): Alist of item IDs corresponding to each item that is desired to be deleted.
        """
        if ids:
            for uuid in ids:
                self.vectorstore.data.delete_by_id(uuid)

    
    ## ----- Semantic Search & Hybrid Search ----- ##

    def similarity_search(self, query: str, n_results: int = 5) -> list:
        """
        Perform a similarity search on collection.

        Parameters:
            query (str): The search query provided by the user.
            n_results (int): The number of items to retrieve in the search results.


        Returns:
            list: A list of search results with relevant documents.
        """
        rs = self.vectorstore.query.near_text(
            query=query, limit=n_results, target_vector="content_vector"
        )

        return self._format_search_results(rs)

    def hybrid_similarity_search(
        self, query: str, n_results: int = 5
    ) -> list:
        """
        Perform a hybrid similarity search on collection.

        Parameters:
            query (str): The search query provided by the user.
            n_results (int): The number of items to retrieve in the search results.
            s_filter (dict): Search filter to reduce search scope (by type, src_id, user..)
            auto_limit (int): The number to limit results based on discontinuities in the result set (read: https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut). 

        Returns:
            list: A list of search results with relevant documents.
        """
        rs = self.vectorstore.query.hybrid(
            query=query,
            limit=n_results,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            target_vector="content_vector"
        )

        return self._format_search_results(rs)