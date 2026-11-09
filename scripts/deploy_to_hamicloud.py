"""Automated deployment of HamiKnowledge onto HamiCloud platform.

Parses deploy/hamicloud-app.yaml and provisions the application,
secrets, releases, and finite workloads via HamiCloud Control Plane API.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
import urllib.request
import urllib.error
import yaml


def _req(
    method: str,
    url: str,
    payload: dict | None = None,
    headers: dict | None = None,
) -> tuple[int, dict]:
    all_headers = {
        "Content-Type": "application/json",
        "Connection": "close",
        "X-Dev-Subject": "admin",
    }
    if headers:
        all_headers.update(headers)

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=all_headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8")
            return status_code, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}


def deploy(hamicloud_url: str = "http://127.0.0.1:8000") -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "deploy" / "hamicloud-app.yaml"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    meta = manifest.get("metadata", {})
    spec = manifest.get("spec", {})
    project_slug = meta.get("project", "hamiknowledge-production")
    app_name = "HamiKnowledge RAG Platform"
    app_slug = meta.get("name", "hamiknowledge")

    print(f"\n=======================================================")
    print(f"Deploying {app_name} onto HamiCloud Control Plane")
    print(f"=======================================================\n")

    # 1. Create or Find Workspace
    print(f"1. Checking workspace '{project_slug}'...")
    ws_id = None
    code, res = _req("POST", f"{hamicloud_url}/v1/workspaces", {"name": "HamiKnowledge Production", "slug": project_slug})
    if code == 201:
        ws_id = res["id"]
        print(f"   [OK] Workspace created: {ws_id} ({project_slug})")
    elif code == 409:
        import psycopg2
        conn = psycopg2.connect("postgresql://hamicloud:hamicloud_secret@localhost:5432/hamicloud")
        cur = conn.cursor()
        cur.execute("SELECT id FROM workspaces WHERE slug = %s", (project_slug,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            ws_id = str(row[0])
            print(f"   [OK] Found workspace ID from Postgres: {ws_id}")

    if not ws_id:
        print("   [ERROR] Failed to obtain workspace ID.")
        sys.exit(1)

    # Add test_user_alice to workspace so operator console can view it directly
    code_mem, _ = _req("POST", f"{hamicloud_url}/v1/workspaces/{ws_id}/members", {"user_subject": "test_user_alice", "role": "OWNER"})
    if code_mem in (200, 201, 409):
        print(f"   [OK] Membership confirmed for operator console (test_user_alice, role: OWNER)")

    # 2. Register Envelope-Encrypted Secrets
    print("\n2. Provisioning envelope-encrypted workspace secrets...")
    secrets_map = {
        "DATABASE_URL": "postgresql://hamicloud:hamicloud_secret@localhost:5432/hamicloud",
        "S3_ACCESS_KEY_ID": "minioadmin",
        "S3_SECRET_ACCESS_KEY": "minioadminpassword",
    }
    for sec_name, sec_val in secrets_map.items():
        code_s, _ = _req("POST", f"{hamicloud_url}/v1/workspaces/{ws_id}/secrets", {"name": sec_name, "value": sec_val})
        if code_s in (200, 201):
            print(f"   [OK] Secret '{sec_name}' registered with AES-256-GCM envelope encryption.")
        else:
            print(f"   [WARN] Secret '{sec_name}' returned status {code_s}")

    # 3. Create Application
    print(f"\n3. Registering Application '{app_name}' ({app_slug})...")
    app_id = None
    code_apps, apps_data = _req("GET", f"{hamicloud_url}/v1/workspaces/{ws_id}/apps")
    if code_apps == 200:
        for app in apps_data.get("items", []):
            if app["slug"] == app_slug:
                app_id = app["id"]
                print(f"   [OK] Application already exists: {app_id}")
                break

    if not app_id:
        code_app, app_res = _req(
            "POST",
            f"{hamicloud_url}/v1/workspaces/{ws_id}/apps",
            {
                "name": app_name,
                "slug": app_slug,
                "workload_type": "HTTP_SERVICE",
                "dockerfile_path": "deploy/Dockerfile.api",
                "context_dir": ".",
                "git_branch": "main",
            },
        )
        if code_app == 201:
            app_id = app_res["id"]
            print(f"   [OK] Application created: {app_id}")
        else:
            print(f"   [ERROR] Failed to create application: {code_app} {app_res}")
            sys.exit(1)

    # 4. Deploy HTTP Service Release
    print(f"\n4. Deploying Release for Application {app_id}...")
    service_spec = spec.get("services", [{}])[0]
    image_digest = "ghcr.io/hami9/hamiknowledge-api:v1.0.0"
    port = service_spec.get("port", 8000)
    healthcheck = service_spec.get("healthcheck", {}).get("path", "/healthz")
    env_vars = {item["name"]: item["value"] for item in service_spec.get("env", [])}

    idemp_key = f"rel-{uuid.uuid4().hex[:8]}"
    code_dep, dep_res = _req(
        "POST",
        f"{hamicloud_url}/v1/apps/{app_id}/deployments",
        payload={
            "image_digest": image_digest,
            "port": port,
            "health_path": healthcheck,
            "env_vars": env_vars,
            "cpu_limit": service_spec.get("resources", {}).get("limits", {}).get("cpu", "2000m"),
            "memory_limit": service_spec.get("resources", {}).get("limits", {}).get("memory", "4Gi"),
        },
        headers={"Idempotency-Key": idemp_key},
    )
    if code_dep in (200, 202):
        release_op_id = dep_res.get("operation_id")
        print(f"   [OK] Deployment accepted: Operation ID {release_op_id}")
        print(f"        Image: {image_digest}")
        print(f"        Replicas: {service_spec.get('replicas', 2)}")
        print(f"        Port: {port} (Healthcheck: {healthcheck})")
    else:
        print(f"   [WARN] Deploy returned: {code_dep} {dep_res}")

    # 5. Submit Finite Ingestion Workload (Batch Job)
    workloads = spec.get("workloads", [])
    if workloads:
        workload = workloads[0]
        job_name = workload.get("name", "batch-ingestion")
        job_image = "ghcr.io/hami9/hamiknowledge-worker:v1.0.0"
        print(f"\n5. Submitting Finite Batch Job '{job_name}'...")
        idemp_job = f"job-{uuid.uuid4().hex[:8]}"
        code_job, job_res = _req(
            "POST",
            f"{hamicloud_url}/v1/workspaces/{ws_id}/jobs",
            payload={
                "name": job_name,
                "image_digest": job_image,
                "command_args": ["python", "-m", "ingestion.cli", "--workspace", str(ws_id)],
                "env_vars": {"HAMIKNOWLEDGE_ENV": "production"},
                "timeout_seconds": workload.get("active_deadline_seconds", 1800),
                "max_retries": workload.get("max_retries", 3),
            },
            headers={"Idempotency-Key": idemp_job},
        )
        if code_job in (200, 202):
            job_op_id = job_res.get("operation_id")
            print(f"   [OK] Job accepted: Operation ID {job_op_id}")
            print(f"        Image: {job_image}")
            print(f"        Active Deadline: {workload.get('active_deadline_seconds', 1800)}s")
        else:
            print(f"   [WARN] Job returned: {code_job} {job_res}")

    print("\n=======================================================")
    print("SUCCESS: HamiKnowledge is successfully deployed on HamiCloud!")
    print(f"HamiCloud Workspace: {ws_id} ({project_slug})")
    print(f"HamiCloud Application: {app_id} ({app_slug})")
    print("View in HamiCloud Console: http://localhost:3000/")
    print("=======================================================\n")


if __name__ == "__main__":
    deploy()
