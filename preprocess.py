import glob
import os
import re
import sys


def read_documents(dirpath: str) -> dict[str, str]:
    files = glob.glob(os.path.join(dirpath, "*_C*.txt"))
    documents = {}
    for filename in files:
        version = filename.replace(".txt", "").split("_")[-1]
        with open(filename, "r", encoding="utf-8") as f:
            documents[version] = f.read()

    return documents


def preprocess_text(text: str) -> str:
    # hack
    text = text.replace("( Template:Lang-tfn )", "")
    text = text.replace("[T]", "T")

    # lowercase
    text = text.lower()

    # normalize newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text.replace("\n", " ")

    # replace tabs with space
    text.replace("\t", " ")

    # replace brackets and slashes [] () \/ with space
    bracket_expr = re.compile(r"[\[\]()\\/]")
    text = bracket_expr.sub(" ", text)

    # remove colon, semicolon, comma, quote " '
    quote_expr = re.compile(r"[\"\':;,]")
    text = quote_expr.sub("", text)

    # remove dots, but keep it in fractions and version number
    dot_expr = re.compile(r"\.(?!\d)")
    text = dot_expr.sub("", text)

    # remove NBSP
    text = text.replace("\u00A0", "")

    # replace newlines with space
    text = text.replace("\n", " ")

    # trim extra spaces
    space_expr = re.compile(r" {2,}")
    text = space_expr.sub(" ", text)

    return text


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, "*_C*.txt"))
    for file_path in files:
        file_name = os.path.basename(file_path)

        print("Processing", file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = preprocess_text(text)
            with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()