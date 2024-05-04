import argparse
from pydub import AudioSegment

def arg_parse():
    # Argument parser
    parser = argparse.ArgumentParser(description="Mix voice and instrument audio")
    parser.add_argument("--vocal_path", type=str, help="Vocal audio path", default="/home/choi/desktop/rvc/ai/data/user2/output/cover/output.wav")
    parser.add_argument("--instrument_path", type=str, help="Instrument audio path", default="/home/choi/desktop/rvc/ai/data/user2/output/music/instrument_origin_music.mp3.wav")
    parser.add_argument("--output_path", type=str, help="Output audio path", default="/home/choi/desktop/rvc/ai/data/user2/output/cover/output_final.wav")

    args = parser.parse_args()
    return args

def main(args):
    # Load vocal and instrument audio
    vocal = AudioSegment.from_file(args.vocal_path)
    instrument = AudioSegment.from_file(args.instrument_path)

    # Mix audio
    mixed = vocal.overlay(instrument)

    # Export mixed audio to output path
    mixed.export(args.output_path, format='wav')

if __name__ == "__main__":
    args = arg_parse()
    main(args)
