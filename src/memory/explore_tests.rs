    use super::choose_note_index;

    #[test]
    fn ripe_note_precedes_older_unripe_note() {
        assert_eq!(choose_note_index(&[0, 3]), 1);
    }

    #[test]
    fn oldest_of_two_ripe_notes_is_selected() {
        assert_eq!(choose_note_index(&[3, 4]), 0);
    }

    #[test]
    fn oldest_note_remains_first_when_none_are_ripe() {
        assert_eq!(choose_note_index(&[0, 2]), 0);
    }
