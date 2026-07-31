MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
    '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
    ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
    '"': '.-..-.', '$': '...-..-', '@': '.--.-.', ' ': '/'
}

REVERSE_MORSE = {v: k for k, v in MORSE_CODE_DICT.items()}


def text_to_morse(text):
    text = text.upper()
    morse = []
    for char in text:
        if char in MORSE_CODE_DICT:
            morse.append(MORSE_CODE_DICT[char])
        else:
            morse.append('?')  # unknown character
    return ' '.join(morse)


def morse_to_text(morse):
    # Words are separated by ' / ', letters by ' '
    words = morse.strip().split(' / ')
    result = []
    for word in words:
        letters = word.split()
        decoded_word = ''
        for code in letters:
            if code in REVERSE_MORSE:
                decoded_word += REVERSE_MORSE[code]
            else:
                decoded_word += '?'
        result.append(decoded_word)
    return ' '.join(result)


def main():
    print("=" * 40)
    print("       MORSE CODE TRANSLATOR")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("  1. Text  → Morse Code")
        print("  2. Morse → Text")
        print("  3. Exit")

        choice = input("\nChoose (1/2/3): ").strip()

        if choice == '1':
            text = input("Enter text: ")
            print(f"Morse Code: {text_to_morse(text)}")

        elif choice == '2':
            print("(Separate letters with spaces, words with ' / ')")
            morse = input("Enter Morse code: ")
            print(f"Decoded Text: {morse_to_text(morse)}")

        elif choice == '3':
            print("Bye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()