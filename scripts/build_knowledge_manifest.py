import json
from pathlib import Path
import logging
from collections import defaultdict

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_manifest(knowledge_file_path: Path, output_path: Path):
    """
    Analyzes a knowledge base file to create a manifest of its structure.

    The manifest contains:
    - A list of all unique categories.
    - A mapping of each category to its available metadata filter fields.
    """
    logging.info(f"Starting manifest generation for: {knowledge_file_path}")

    if not knowledge_file_path.exists():
        logging.error(f"Knowledge file not found at: {knowledge_file_path}")
        return

    with open(knowledge_file_path, 'r', encoding='utf-8') as f:
        knowledge_data = json.load(f)

    # Corrected: Access the list of items under the 'qa_pairs' key
    items_list = knowledge_data.get('qa_pairs')
    if not items_list or not isinstance(items_list, list):
        logging.error("JSON structure is not as expected. 'qa_pairs' key not found or is not a list.")
        return

    categories = set()
    metadata_fields = defaultdict(set)

    for item in items_list:
        # The 'category' is directly in the item, but other potential metadata might be nested.
        # For this knowledge base, metadata is simple. We'll check 'category' and 'tags'.
        category = item.get('category')
        if category:
            categories.add(category)
            # Example of extracting more metadata fields if they existed in a 'metadata' dict
            # if 'metadata' in item and isinstance(item['metadata'], dict):
            #     for key in item['metadata'].keys():
            #         metadata_fields[category].add(key)
            
            # For this specific file, we see 'tags' can be considered a filterable field.
            if 'tags' in item:
                 metadata_fields[category].add('tags')

        else:
            logging.warning(f"Item with id '{item.get('id')}' is missing a 'category'.")

    # Convert sets to sorted lists for stable output
    manifest = {
        'categories': sorted(list(categories)),
        'metadata_fields_by_category': {
            cat: sorted(list(fields)) for cat, fields in metadata_fields.items()
        }
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    logging.info(f"Successfully created knowledge manifest at: {output_path}")
    logging.info(f"Found {len(manifest['categories'])} categories.")
    logging.info("Metadata fields per category:")
    for cat, fields in manifest['metadata_fields_by_category'].items():
        logging.info(f"  - {cat}: {fields}")


if __name__ == '__main__':
    # Define paths relative to the project root
    ROOT_DIR = Path(__file__).parent.parent
    KNOWLEDGE_FILE = ROOT_DIR / 'data' / '云和文旅知识库数据集.json'
    MANIFEST_FILE = ROOT_DIR / 'data' / 'knowledge_manifest.json'

    build_manifest(KNOWLEDGE_FILE, MANIFEST_FILE) 