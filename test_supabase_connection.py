import os
from supabase_client import get_supabase

sb = get_supabase()
resp = sb.table("leads").select("lead_id").limit(1).execute()
print("✅ Connected to Supabase.")
print("Sample result:", resp.data)
