CREATE DATABASE duplicate_manager;

USE duplicate_manager;

CREATE TABLE scans (
    scan_id INT AUTO_INCREMENT PRIMARY KEY,
    scan_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    directory TEXT
);

CREATE TABLE files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    scan_id INT,
    file_path TEXT,
    checksum VARCHAR(100),
    size BIGINT,
    FOREIGN KEY (scan_id) REFERENCES scans(scan_id)
);

CREATE TABLE deleted_files (
    del_id INT AUTO_INCREMENT PRIMARY KEY,
    file_path TEXT,
    deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);