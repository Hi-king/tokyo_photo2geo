PROJECT_ID=hiking-327807
APP_NAME=photo2geo
docker build -t gcr.io/${PROJECT_ID}/${APP_NAME}:latest .
docker push gcr.io/${PROJECT_ID}/${APP_NAME}:latest
