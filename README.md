docker run --name frasier-bot-local-db \
  -e POSTGRES_USER=frasier_bot_admin \
  -e POSTGRES_PASSWORD=password123 \
  -e POSTGRES_DB=frasier_bot \
  -p 5432:5432 \
  -v frasier-bot_pgdata:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16