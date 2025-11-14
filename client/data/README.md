This is a critical step:

This project uses a "Bring Your Own Data" model. The data is partitioned and must be placed here manually.

Go to your Google Drive: Find the /FL_Project/client_data/ folder.

Download all 5 .npz files.

Deployment Instructions:

Server Person:

You MUST keep global_test_set.npz in this folder.

You can delete all other client_... files.

Run python server.py.

Client 1 ("hospital"):

You MUST keep client_hospital_train.npz and client_hospital_test.npz.

You can delete all other files.

Run python run_client.py --client-id hospital.

Client 2 ("factory"):

You MUST keep client_factory_train.npz and client_factory_test.npz.

You can delete all other files.

Run python run_client.py --client-id factory.