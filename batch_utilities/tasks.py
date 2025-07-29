from batch_utilities.celery_app import celery
from batch_utilities.product_sync import process_products, process_batch

@celery.task
def sync_products_task(shop, access_token):
    print(f"Starting sync for {shop}")
    batch_id, base_dir, items = process_products(shop, access_token)
    process_batch(base_dir, batch_id, shop, items)
    print(f"Sync complete for {shop}")
