# Self RAG

Personal Markdown RAG learning console.

## Run On A Server

Start the backend:

```bash
bash scripts/dev-backend.sh
```

Start the frontend:

```bash
bash scripts/dev-frontend.sh
```

Then open the frontend from your machine:

```text
http://<server-ip>:5173/
```

FastAPI docs are available at:

```text
http://<server-ip>:8800/docs
```

The frontend calls backend APIs through the Vite proxy, so the browser only needs access to port `5173`. The server itself must be able to reach `127.0.0.1:8800`.

