import secrets
import numpy as np

#############################################
## This class is for implementing a simple ##
## Information-Theoretic MAC               ##
#############################################


# Function for generating a generator for the MAC
# @param prime: The prime numper to be used for the MAC
# @return: The generator for the MAC on the form key=(alpha, beta)
def generate_mac_key(prime):
    # Generate two random numbers:
    alpha, beta = secrets.randbelow(prime), secrets.randbelow(prime)
    # Define the key:
    key = (alpha, beta)
    # Output the key
    return key


# function for generating a homomorphic MAC key
# @param prime: The prime number to be used for the MAC
# @param key_suffix: The part of the key that is different for each value to be MAC'ed
# @return: The generator for the MAC on the form key_i=(alpha, beta_i) for different key values
def generate_homomorphic_mac_key(prime, key_suffix):
    # Generate random numbers:
    alpha = secrets.randbelow(prime)
    beta_i = secrets.randbelow(key_suffix)
    # Define the key:
    key_i = (alpha, beta_i)
    # Output the key
    return key_i


# Function for generating a Tag "a signing for the MAC"
# @param key: The key for the MAC
# @param message: The message to be signed
# @return: The tag for the message
def generate_mac_tag(key, message):
    # Extract the key
    alpha, beta = key
    # Calculate the tag (Do we need to modulo the prime?)
    tag = (alpha * message + beta)
    # Output the tag
    return tag


# Function for generating a homomorphic MAC tag
# @param key_suffix: The different keys for the MAC
# @param message_suffix: The different messages to be signed
# @return: The tag for the message_suffix
def generate_homomorphic_mac_tag(key_suffix, message_suffix):
    # Extract the key
    alpha, beta_i = key_suffix
    # Calculate the tag (Do we need to modulo the prime?)
    tag_i = (alpha * message_suffix + beta_i)


# Function for verifying the tag
# @param key: The key for the MAC
# @param tag: The tag to be verified
# @param message: The message to be verified
# @return: True if the tag is valid, False otherwise
def verify_mac_tag(key, tag, message):
    # Extract the key
    alpha, beta = key
    # Calculate the tag
    calculated_tag = (alpha * message + beta)
    # Check if the tags are equal
    if calculated_tag == tag:
        return True
    else:
        return False


# Function for verifying a homomorphic MAC tag
# @param tags: All the tags of the messages
# @param keys: All the keys of the messages
# @param messages: All the messages
# @return: True if the tags are valid, False otherwise
def verify_homomorphic_mac_tag(tags, keys, messages):
    # Initialize the calculated tags
    calculated_tags = []
    # Iterate through the messages
    for i in range(len(messages)):
        # Extract the key
        alpha, beta_i = keys[i]
        # Calculate the tag
        calculated_tag = (alpha * messages[i] + beta_i)
        # Append the calculated tag
        calculated_tags.append(calculated_tag)
    # Check if the tags are equal
    if calculated_tags == tags:
        return True
    else:
        return False