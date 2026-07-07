from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import botocore


def get_aws_service_client(
    service_name: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    profile_name: Optional[str] = None,
    max_retries: Optional[int] = 3,
    timeout: Optional[float] = 60.0,
) -> "botocore.client.BaseClient":
    """
    Create a boto3 client for the given AWS service.

    If ``aws_access_key_id`` is provided (and ``profile_name`` is not), the client
    is created from an explicit key/secret/session-token session. Otherwise, a
    session is created from ``profile_name`` (which may be ``None`` to fall back
    to boto3's default credential resolution, e.g. environment variables or
    the instance/task role).

    Args:
        service_name (Optional[str]): The name of the AWS service to create a client
            for (e.g. "s3", "bedrock-runtime").
        region_name (Optional[str]): The AWS region to use.
        aws_access_key_id (Optional[str]): AWS access key id for explicit credentials.
        aws_secret_access_key (Optional[str]): AWS secret access key for explicit
            credentials.
        aws_session_token (Optional[str]): AWS session token for explicit,
            temporary credentials.
        profile_name (Optional[str]): Name of the AWS credentials profile to use.
        max_retries (Optional[int]): Maximum number of retries for failed requests.
        timeout (Optional[float]): Connection timeout, in seconds.

    Returns:
        botocore.client.BaseClient: The configured boto3 service client.

    Raises:
        ImportError: If ``boto3`` and ``botocore`` are not installed.
        ValueError: If the client cannot be created with the provided credentials.

    """
    try:
        import boto3
        import botocore
    except ImportError:
        raise ImportError(
            "Please run `pip install boto3 botocore` to use AWS services."
        )

    config = botocore.config.Config(
        retries={"max_attempts": max_retries or 0, "mode": "standard"},
        connect_timeout=timeout,
    )

    try:
        if not profile_name and aws_access_key_id:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            client = session.client(service_name, config=config)  # type: ignore
        else:
            session = boto3.Session(profile_name=profile_name)
            if region_name:
                client = session.client(
                    service_name,
                    region_name=region_name,
                    config=config,  # type: ignore
                )
            else:
                client = session.client(service_name, config=config)  # type: ignore
    except Exception as e:
        raise ValueError(
            f"Please verify the provided credentials. Failed to create AWS client "
            f"for service '{service_name}': {e}"
        ) from e

    return client
