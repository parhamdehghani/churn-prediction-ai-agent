FROM astrocrpublic.azurecr.io/runtime:3.0-8

# Switch to the root user to install system packages
USER root

# Add prerequisites and the Google Cloud SDK repository
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update -y

# Switch back to the astro user before letting the standard Astro entrypoint take over
USER astro
