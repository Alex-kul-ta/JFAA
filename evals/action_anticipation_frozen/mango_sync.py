import os
import ssl

from irods.session import iRODSSession
from mango_auth import iinit

IRODS_BASE_PATH = (
    "/set/home/ciis-lab/experiments/JFAA"
)

def irods_connections():
    """
    Establishes a connection to the iRODS server using the iRODS session.
    Returns the iRODS session object.
    """
    # Initialize iRODS authentication
    iinit("u0172623", "set", "set.irods.icts.kuleuven.be")

    # Create an iRODS session
    try:
        try:
            env_file = os.environ["IRODS_ENVIRONMENT_FILE"]
        except KeyError:
            env_file = os.path.expanduser("~/.irods/irods_environment.json")

        ssl_context = ssl.create_default_context(
            purpose=ssl.Purpose.SERVER_AUTH,
            cafile=None,
            capath=None,
            cadata=None,
        )
        ssl_settings = {"ssl_context": ssl_context}

    except Exception as e:
        raise RuntimeError(f"Failed to configure iRODS connection: {e}")

    session = iRODSSession(irods_env_file=env_file, **ssl_settings)
    return session

def sync_experiments_to_mango(local_experiments_dir, irods_base_path=IRODS_BASE_PATH):
    """
    Syncs the local experiments directory to the iRODS server (Mango).

    Continues uploading when an individual file fails, then raises an error
    containing all failed paths after the session has been cleaned up.
    """
    session = irods_connections()
    failed_uploads = []

    try:
        # Ensure the target directory exists in iRODS
        irods_experiments_path = os.path.join(irods_base_path, os.path.basename(local_experiments_dir))
        if not session.collections.exists(irods_experiments_path):
            session.collections.create(irods_experiments_path)

        # Sync files from local to iRODS
        for root, dirs, files in os.walk(local_experiments_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_experiments_dir)
                irods_file_path = os.path.join(irods_experiments_path, relative_path)

                try:
                    # Create necessary collections in iRODS
                    irods_collection_path = os.path.dirname(irods_file_path)
                    if not session.collections.exists(irods_collection_path):
                        session.collections.create(irods_collection_path)

                    # Upload the file to iRODS
                    if not session.data_objects.exists(irods_file_path):
                        with open(local_file_path, "rb") as file_handle:
                            session.data_objects.put(file_handle, irods_file_path)
                    else:
                        remote_file = session.data_objects.get(irods_file_path)
                        local_modified_time = os.path.getmtime(local_file_path)
                        if local_modified_time > remote_file.modify_time.timestamp():
                            with open(local_file_path, "rb") as file_handle:
                                session.data_objects.put(file_handle, irods_file_path)
                except Exception as exc:
                    failed_uploads.append((local_file_path, exc))
    finally:
        session.cleanup()

    if failed_uploads:
        failed_paths = ", ".join(path for path, _ in failed_uploads)
        raise RuntimeError(f"Failed to sync {len(failed_uploads)} file(s): {failed_paths}")