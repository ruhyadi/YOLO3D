# Execute development docker container
echo "Executing docker container"
docker compose -f docker-compose.trainer.yaml down -t 0
docker compose -f docker-compose.trainer.yaml up -d\
    && docker exec -it yolo3d-trainer bash\
    && 
echo "Docker container execution complete"