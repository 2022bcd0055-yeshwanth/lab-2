pipeline {
    agent any

    environment {
        IMAGE = "2022bcd0055yeshwanth/2022bcd0055-ml:latest"
        CONTAINER = "ml_infer_test"
        PORT = "8000"
    }

    stages {

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
                for i in {1..10}
                do
                  sleep 5
                  curl -f http://localhost:8000/docs && break
                done
                '''
            }
        }

        stage('Valid Inference Test') {
            steps {
                sh '''
                QUERY=$(jq -r 'to_entries|map("\\(.key)=\\(.value)")|join("&")' valid_input.json)

                curl "http://localhost:8000/predict?$QUERY" > valid_output.json
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
                "http://localhost:8000/predict?$QUERY")

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
