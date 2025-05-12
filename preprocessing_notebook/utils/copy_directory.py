#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

def copy_directory(source_dir, dest_dir):
    """
    Copy all files from source_dir to dest_dir.
    If dest_dir doesn't exist, it will be created.
    If dest_dir exists, it will be overwritten.
    
    Args:
        source_dir (str): Path to the source directory
        dest_dir (str): Path to the destination directory
    """
    # Check if source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist")
    
    # If destination directory exists, remove it to overwrite
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    # Copy the directory tree
    shutil.copytree(source_dir, dest_dir)
    print(f"Successfully copied all files from '{source_dir}' to '{dest_dir}'")

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Copy all files from one directory to another')
    parser.add_argument('source_dir', help='Source directory path')
    parser.add_argument('dest_dir', help='Destination directory path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the copy function
    copy_directory(args.source_dir, args.dest_dir)

if __name__ == "__main__":
    main()