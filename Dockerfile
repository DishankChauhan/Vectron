FROM rust:1.82 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /vectron

# Copy the Rust project
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the project
RUN cargo build --release --bin vectron

FROM python:3.11-slim

# Install dependencies for Python
RUN pip install --no-cache-dir sentence-transformers

# Copy Python embedding script
WORKDIR /vectron
COPY python ./python

# Copy the compiled binary from the builder stage
COPY --from=builder /vectron/target/release/vectron /usr/local/bin/vectron

# Create data directory for persistence
RUN mkdir -p /vectron/data

# Expose the port the server listens on
EXPOSE 3000

# Run the server
CMD ["vectron"] 