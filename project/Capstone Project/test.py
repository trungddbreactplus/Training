import os
from dotenv import load_dotenv
from zep_cloud.client import Zep
from zep_cloud.types import Message
from pprintpp import pprint

load_dotenv()

ZEP_API = os.getenv("ZEP_API")
zep_client = Zep(api_key=ZEP_API)

# thread = zep_client.thread.get(thread_id='ef2b0392-e808-4b66-88be-d7b381525b6e')
# pprint(thread.schema())
query = 'Tôi tên gì'

# zep_client.context.create_context_template(
#     template_id="customer-support",
#     template="""# CUSTOMER PROFILE
# %{user_summary}
# # FACTS
# %{edges limit=10}
# # KEY ENTITIES
# %{entities limit=5}"""
# )
print(zep_client.context.list_context_templates())

user_context = zep_client.thread.get_user_context(
    thread_id="cc90cf57-59a8-4354-aff9-be16421ad3bf",
    template_id="customer-support"
)
context_block = user_context.context
pprint(context_block)