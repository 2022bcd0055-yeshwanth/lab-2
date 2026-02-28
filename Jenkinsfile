pipeline {
    agent any

    environment {
        IMAGE = "2022bcd0055yeshwanth/2022bcd0055-ml:latest"
        CONTAINER = "ml_infer_test"
        PORT = "8000"
        BASE_URL = "http://host.docker.internal:8000"
    }

    stages {

        stage('Cleanup Old Container') {
            steps {
                sh 'docker rm -f $CONTAINER || true'
            }
        }

        stage('Pull Docker Image') {
            steps {
                sh 'docker pull $IMAGE'
            }
        }

        stage('Run Container') {
            steps {
                sh '''
                docker run -d -p 8000:8000 --name $CONTAINER $IMAGE
                '''
            }
        }

        stage('Wait for API') {
            steps {
                sh '''
                for i in {1..12}
                do
                  sleep 5
                  curl -f $BASE_URL/docs && exit 0
                done
                echo "API not ready"
                exit 1
                '''
            }
        }

        stage('Install jq') {
            steps {
                sh '''
                apt-get update
                apt-get install -y jq
                '''
            }
        }

        stage('Valid Inference Test') {
            steps {
                sh '''
                QUERY=$(jq -r 'to_entries|map("\\(.key)=\\(.value)")|join("&")' valid_input.json)

                curl "$BASE_URL/predict?$QUERY" > valid_output.json
                '''

                sh 'cat valid_output.json'

                sh 'grep -q wine_quality valid_output.json'
            }
        }

        stage('Invalid Inference Test') {
            steps {
                sh '''
                QUERY=$(jq -r 'to_entries|map("\\(.key)=\\(.value)")|join("&")' invalid_input.json)

                STATUS=$(curl -s -o invalid_output.json -w "%{http_code}" \
                "$BASE_URL/predict?$QUERY")

                if [ "$STATUS" -eq 200 ]; then
                  echo "Invalid test failed"
                  exit 1
                fi
                '''

                sh 'cat invalid_output.json'
            }
        }

        stage('Stop Container') {
            steps {
                sh '''
                docker stop $CONTAINER
                docker rm $CONTAINER
                '''
            }
        }
    }
}
