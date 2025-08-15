pipeline {
    agent any

    environment {
        // Define variables used across stages
        IMAGE_NAME = "churn-prediction-api"
        GCP_PROJECT_ID = "churn-prediction-ai-agent" 
        GCP_REGION = "us-central1"
        REPO_NAME = "churn-repo"
        IMAGE_TAG = "${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code from Git...'
                checkout scm
            }
        }

        /* stage('Pull DVC Data') {
    steps {
        withCredentials([file(credentialsId: 'gcp-service-account-key', variable: 'GCP_KEY_FILE')]) {
            // Set the environment variable for DVC and other Python libraries
            sh """
            export GOOGLE_APPLICATION_CREDENTIALS=${GCP_KEY_FILE}
            echo 'Pulling data with DVC...'
            dvc pull --force
               """
        }
    }
} */

        stage('Run Tests') {
            steps {
                // Future tests
                echo 'Skipping tests for now.'
            }
        }

        stage('Build & Push Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_TAG}"
                // Use the Google Service Account credential
                withCredentials([file(credentialsId: 'gcp-service-account-key', variable: 'GCP_KEY_FILE')]) {
                    // Authenticate gcloud CLI
                    sh "gcloud auth activate-service-account --key-file=${GCP_KEY_FILE}"
                    
                    // Configure Docker to authenticate with Artifact Registry
                    sh "gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev --quiet"
                    
                    // Build and push multi-platform Docker image
                    sh """
   			 docker buildx build \
      			--platform linux/amd64,linux/arm64 \
      			--tag ${IMAGE_TAG} \
      			--push \
      			.
			"""
                }
            }
        }
    }
}
