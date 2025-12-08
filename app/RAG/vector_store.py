from app.config import settings
from loguru import logger
from pymilvus import connections, db, exceptions, utility, Collection
from fastapi import HTTPException, status

class MilvusConnectionError(Exception):
    pass

class MilvusCollectionError(Exception):
    pass

def connect_to_milvus(create_db: bool = False) -> bool:
    try:
        connections.connect(
            alias=settings.MILVUS_ALIAS,
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN
        )
        if create_db:
            create_database()
        db.using_database(settings.MILVUS_DB_NAME)
        return True
    except exceptions.MilvusException as e:
        error_msg = f"Failed to connect to Milvus at {settings.MILVUS_URI}: {str(e)}"
        logger.error(error_msg)
        raise MilvusConnectionError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error connecting to Milvus: {str(e)}"
        logger.error(error_msg)
        raise MilvusConnectionError(error_msg) from e

def create_database() -> bool:
    try:
        actual_db_name = settings.MILVUS_DB_NAME
        databases = db.list_database()
        if actual_db_name not in databases:
            logger.info(f"Database '{actual_db_name}' not found. Creating it.")
            db.create_database(actual_db_name)
            logger.info(f"Successfully Created Database '{actual_db_name}'")
        else:
            logger.info(f"Database '{actual_db_name}' already exists.")
        return True
    except exceptions.MilvusException as e:
        error_msg = f"Failed to create database '{actual_db_name}': {str(e)}"
        logger.error(error_msg)
        raise MilvusConnectionError(error_msg) from e

def load_all_milvus_collections():
    try:
        collections = utility.list_collections()
        logger.info(f"Found collections: {collections}")
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return

    for name in collections:
        try:
            collection = Collection(name)
            collection.load()
            logger.info(f"Loaded collection: {name}")
        except Exception as e:
            logger.error(f"Failed to load collection '{name}': {e}")

def unload_all_milvus_collections():
    try:
        collections = utility.list_collections()
        logger.info(f"Found collections to unload: {collections}")
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return

    for name in collections:
        try:
            collection = Collection(name)
            collection.flush()
            collection.release()
            logger.info(f"Unloaded (released) collection: {name}")
        except Exception as e:
            logger.error(f"Failed to unload collection '{name}': {e}")
    try:
        utility.flush_all()
        logger.info("All collections from current database flushed and released")
    except Exception as e:
        logger.error(f"Failed to flush all collections: {e}")

def disconnect_from_milvus() -> bool:
    try:
        # Now disconnect
        connections.disconnect(alias=settings.MILVUS_ALIAS)
        logger.info(f"Disconnected from Milvus (alias: {settings.MILVUS_ALIAS})")
        return True
    except exceptions.MilvusException as e:
        error_msg = f"Failed to disconnect from Milvus (alias: {settings.MILVUS_ALIAS}): {str(e)}"
        logger.error(error_msg)
        raise MilvusConnectionError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error disconnecting from Milvus: {str(e)}"
        logger.error(error_msg)
        raise MilvusConnectionError(error_msg) from e