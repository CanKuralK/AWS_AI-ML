"""
SerializeImageData:  Lambda function to serialize the image data
"""
import json
import boto3
import base64

# A low-level client representing Amazon Simple Storage Service (S3)
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input (You may also check lambda test)
    key = event['s3_key']                              ## TODO: fill in
    bucket = event['s3_bucket']              ## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:          # Read binary file
        image_data = base64.b64encode(f.read())      # Base64 encode binary data ('image_data' -> class:bytes)

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    
   
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,      # Bytes data
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

"""
ImageClassification

Takes the image output from the SerializeImageData function, decodes it, and then pass inferences back to the last lambda in Step Function
"""

import json
import base64
import boto3

# Using low-level client representing Amazon SageMaker Runtime ( To invoke endpoint)
runtime_client = boto3.client('sagemaker-runtime')                   


# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-10-23-21-48-30-169'       ## TODO: fill in (Trained IC Model Name)


def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])           ## TODO: fill in (Decoding the encoded 'Base64' image-data and class remains'bytes')

    # Instantiate a Predictor (Here we have renamed 'Predictor' to 'response')
    # Response after invoking a deployed endpoint via SageMaker Runtime 
    response = runtime_client.invoke_endpoint(
                                        EndpointName=ENDPOINT,    # Endpoint Name
                                        Body=image,               # Decoded Image Data as Input (class:'Bytes') Image Data
                                        ContentType='image/png'   # Type of inference input data - Content type (Eliminates the need of serializer)
                                    )
                                    
    
    # Make a prediction: Unpack reponse
    # (NOTE: 'response' returns a dictionary with inferences in the "Body" : (StreamingBody needs to be read) having Content_type='string')
    
    ## TODO: fill in (Read and decode predictions/inferences to 'utf-8' & convert JSON string obj -> Python object)
    inferences = json.loads(response['Body'].read().decode('utf-8'))     # list
  
    
    # We return the data back to the Step Function    
    event['inferences'] = inferences            ## List of predictions               
    return {
        'statusCode': 200,
        'body': event                           ## Passing the event python dictionary in the body
    }

"""
FilterLowConfidenceInferences

Filters low confidence inferences received from the Image Classification function
"""

import json

THRESHOLD = 0.95


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['inferences']            ## TODO: fill in
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(list(inferences))>THRESHOLD           ## TODO: fill in (True, if a value exists above 'THRESHOLD')
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("COFIDENCE_THRESHOLD_NOT_MET")

    return {
        'statusCode': 200,
        'body': event            ## Passing the final event as a python dictionary
    }
