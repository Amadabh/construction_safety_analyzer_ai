import boto3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

def _make_sns_client():
    """Build SNS client — uses system credentials by default, .env creds only as fallback."""
    kwargs = {}
    if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"]     = Config.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = Config.AWS_SECRET_ACCESS_KEY
    return boto3.client("sns", **kwargs)

sns = _make_sns_client()

def initialize_sns(topic_name: str = 'risk_alerts') -> str:
    """Creates topic if not exists and ensures email is subscribed. Returns topic_arn."""
    response = sns.create_topic(Name=topic_name)  # idempotent
    topic_arn = response['TopicArn']
    
    status = _is_email_subscribed(topic_arn, Config.ALERT_EMAIL_DEST)
    if status['subscribed']:
        print(f"{Config.ALERT_EMAIL_DEST} already subscribed: {status['status']}")
    else:
        sns.subscribe(TopicArn=topic_arn, Protocol='email', Endpoint=Config.ALERT_EMAIL_DEST)
        print(f"Subscribed {Config.ALERT_EMAIL_DEST} to {topic_arn}")
    
    return topic_arn

def publish_alert(topic_arn: str, subject: str, message: str) -> bool:
    """Publishes a message to the SNS topic. Returns True if successful."""
    try:
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
        return True
    except Exception as e:
        print(f"SNS publish failed: {e}")
        return False

def _is_email_subscribed(topic_arn: str, email_address: str) -> dict:
    paginator = sns.get_paginator('list_subscriptions_by_topic')
    for response in paginator.paginate(TopicArn=topic_arn):
        for sub in response['Subscriptions']:
            if sub['Endpoint'] == email_address and sub['Protocol'] == 'email':
                return {
                    'subscribed': True,
                    'status': sub['SubscriptionArn'] if sub['SubscriptionArn'] != 'PendingConfirmation' else 'PendingConfirmation'
                }
    return {'subscribed': False, 'status': None}