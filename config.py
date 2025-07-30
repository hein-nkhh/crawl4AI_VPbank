import logging

LOG_LEVEL = logging.DEBUG  # Mức độ log mặc định
AWS_REGION = "ap-southeast-1"  # Khu vực AWS mặc định
S3_BUCKET_NAME = "team253vpbank"  # Tên bucket S3 mặc định
PROMPT_EXTRACTOR_FILE = "prompts/prompt_extractor.txt"  # Đường dẫn tới file prompt extractor
PROMPT_PERSONAL_COMPARE_FILE = './prompts/prompt_compare_new_old_personal_risk_info.txt'
PROMPT_ORGANIZATION_COMPARE_FILE = 'D:/VPBankHackathon/prompts/prompt_compare_new_old_organization.txt'

TABLE_CONFIG = {
        'media_config': {"adverse_media": "media_id"},
        'person_config': {"personal_info": "per_id"},
        'organization_config': {"organization_info": "org_id"},
        'p2m_config': {"personal2media": "p2m_id"},
        'o2m_config': {"org2media": "o2m_id"},
        'article_embedd2media_config': {"article_embedd2media": "article_embedd_id"},
        'personal_embedd2per_config': {"personal_embedd2per": "personal_embedd_id"},
        'org_embedd2org_config': {"org_embedd2org": "org_embedd_id"}
    }

TABLE_CONFIG_DEMO = {
        'media_config': {"adverse_media_demo": "media_id"},
        'person_config': {"personal_info_demo": "per_id"},
        'organization_config': {"organization_info_demo": "org_id"},
        'p2m_config': {"personal2media_demo": "p2m_id"},
        'o2m_config': {"org2media_demo": "o2m_id"},
        'article_embedd2media_config': {"article_embedd2media_demo": "article_embedd_id"},
        'personal_embedd2per_config': {"personal_embedd2media_demo": "personal_embedd_id"},
        'org_embedd2org_config': {"org_embedd2media_demo": "org_embedd_id"},
        'personal_risk_embedd2per_config': {"personal_risk_embedd2per_demo": "personal_risk_embedd_id"}, 
        'organization_risk_embedd2org_config': {"organization_risk_embedd2org_demo": "organization_risk_embedd_id"}
    }


EMBEDDING_MODEL_NAME = 'D:\VPBankHackathon\embedder\models\phobert_base_v2_local'
EMBEDDING_DIM = 1024
AWS_VIRGINA_REGION = 'us-east-1'
DEEPSEEK_MODEL_VIRGINA_ID = "arn:aws:bedrock:us-east-1:538830382271:inference-profile/us.deepseek.r1-v1:0"


CLAUDE_30_HAIKU_ON_DEMAND_VIRGINA_MODEL_ID = ''

CLAUDE_35_HAIKU_CROSS_REGION_VIRGINA_MODEL_ID = 'arn:aws:bedrock:us-east-1:538830382271:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0'
CLAUDE_35_HAIKU_ON_DEMAND_VIRGINA_MODEL_ID = ''

CLAUDE_37_SONNET_CROSS_REGION_MODEL_ID = 'arn:aws:bedrock:us-east-1:538830382271:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0'


USE_MATCHING_METHOD = True

