#!/bin/bash

# CVAT Diagnostics Script
# Checks issues from most likely to least likely

OUTPUT_FILE="cvat_diagnostics_$(date +%Y%m%d_%H%M%S).txt"

echo "CVAT Diagnostics Report - $(date)" > $OUTPUT_FILE
echo "============================================" >> $OUTPUT_FILE

# 1. Most Likely: Services still starting
echo "1. CHECKING CONTAINER STATUS (Most Likely Issue)" >> $OUTPUT_FILE
echo "---------------------------------------------------" >> $OUTPUT_FILE
docker compose ps >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 2. Port conflicts
echo "2. CHECKING PORT CONFLICTS" >> $OUTPUT_FILE
echo "----------------------------" >> $OUTPUT_FILE
echo "Port 8081 usage:" >> $OUTPUT_FILE
sudo lsof -i :8081 >> $OUTPUT_FILE 2>&1
echo "Port 8080 usage:" >> $OUTPUT_FILE
sudo lsof -i :8080 >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 3. Container health and logs
echo "3. CHECKING CONTAINER LOGS" >> $OUTPUT_FILE
echo "---------------------------" >> $OUTPUT_FILE
echo "CVAT Server logs (last 20 lines):" >> $OUTPUT_FILE
docker compose logs --tail=20 cvat_server >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE
echo "Database logs (last 10 lines):" >> $OUTPUT_FILE
docker compose logs --tail=10 cvat_db >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 4. Network connectivity
echo "4. CHECKING NETWORK CONNECTIVITY" >> $OUTPUT_FILE
echo "---------------------------------" >> $OUTPUT_FILE
echo "Docker networks:" >> $OUTPUT_FILE
docker network ls >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE
echo "CVAT network inspect:" >> $OUTPUT_FILE
docker network inspect cvat_default >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 5. Resource usage
echo "5. CHECKING SYSTEM RESOURCES" >> $OUTPUT_FILE
echo "-----------------------------" >> $OUTPUT_FILE
echo "Memory usage:" >> $OUTPUT_FILE
free -h >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE
echo "Disk usage:" >> $OUTPUT_FILE
df -h >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 6. Docker daemon status
echo "6. CHECKING DOCKER DAEMON" >> $OUTPUT_FILE
echo "--------------------------" >> $OUTPUT_FILE
systemctl status docker >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 7. Container inspection (least likely)
echo "7. DETAILED CONTAINER INSPECTION (Least Likely)" >> $OUTPUT_FILE
echo "------------------------------------------------" >> $OUTPUT_FILE
echo "CVAT Server container details:" >> $OUTPUT_FILE
docker inspect cvat_server >> $OUTPUT_FILE 2>&1
echo "" >> $OUTPUT_FILE

# 8. Compose file validation
echo "8. COMPOSE FILE VALIDATION" >> $OUTPUT_FILE
echo "---------------------------" >> $OUTPUT_FILE
docker compose config >> $OUTPUT_FILE 2>&1

echo "" >> $OUTPUT_FILE
echo "Diagnostics complete. Check $OUTPUT_FILE for results." >> $OUTPUT_FILE

echo "Diagnostics saved to: $OUTPUT_FILE"
echo "Most likely issues are at the top of the file."