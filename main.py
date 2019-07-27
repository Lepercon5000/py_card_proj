import json
import logging
import os
import requests

import net
import sys
from argparse import ArgumentParser
from pathlib import Path


root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)

_SCRY_FALL_BULK_DOWNLOAD = 'https://archive.scryfall.com/json/scryfall-default-cards.json'


def db_filepath(download_directory: Path):
    return download_directory / 'scryfall-default-cards.json'


def download_cards_db(download_location: Path):
    response = requests.get(_SCRY_FALL_BULK_DOWNLOAD)
    with download_location.open('wb') as db_file_stream:
        db_file_stream.write(response.content)

    logger.info('finished downloading db file from scryfall')


def read_cards_db(download_location: Path):
    with download_location.open('r', encoding='utf8', errors='ignore') as cards_db_fp:
        cards = json.load(cards_db_fp)

    logger.info('finished reading db file from scryfall')
    return cards


def get_img_uris(cards):
    noncard_types = ['token', 'emblem', 'double_faced_token', 'vanguard']
    for card in cards:
        if card['layout'] not in noncard_types:
            if 'image_uris' in card:
                yield (card['id'], card['image_uris'], card['lang'])
            elif 'card_faces' in card and any(card['card_faces']):
                for card_face in card['card_faces']:
                    yield (str(card['id']) + '_' + card_face['name'], card_face['image_uris'], card['lang'])


def download_img(card_info, download_directory: Path):
    if 'large' in card_info[1]:
        url = card_info[1]['large']
    elif 'border_crop' in card_info[1]:
        url = card_info[1]['border_crop']
    else:
        raise Exception('No large or border_crop!')
    save_location = download_directory / 'cards' / \
        'download' / (card_info[0] + '.jpg')
    if not save_location.exists():
        response = requests.get(url)
        with save_location.open('wb+') as jpg_fp:
            jpg_fp.write(response.content)
            logger.info("Downloaded %s to %s", url, save_location)
    else:
        if card_info[2] != 'en':
            os.remove(str(save_location))
            logger.info("Deleted %s", save_location)
        else:
            logger.info("Skipping %s", url)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('card_store',
                            type=Path,
                            help='Directory used to download card images and image batches')

    arg_parser.add_argument('results',
                            type=Path,
                            help='Directory used to store results, such as the model file and debugged results')

    arg_parser.add_argument('--get_scry_db',
                            action='store_true',
                            help='Download scryfall cards json file to card_store')

    arg_parser.add_argument('--dl_imgs',
                            action='store_true',
                            help='Download card images using scry fall json file')

    arg_parser.add_argument('--batch_size',
                            type=int,
                            default=300,
                            help=('Default assumes ~11Gigabytes of GPU memory.'
                                  'You\'ll want to enter (number of gigabytes on GPU) * 27 as this '
                                  'will allocate the approprate batch size to maximize the GPU'))

    args = arg_parser.parse_args()
    from pprint import pprint

    pprint(args)

    card_store = args.card_store
    db_file_location = db_filepath(card_store)

    if args.get_scry_db:
        download_cards_db(db_file_location)

    if args.dl_imgs:
        cards = read_cards_db(db_file_location)
        for card in get_img_uris(cards):
            download_img(card, card_store)

    net.start_training(
        card_store,
        args.results,
        args.batch_size)

    logger.info('finished training')


if __name__ == "__main__":
    main()
