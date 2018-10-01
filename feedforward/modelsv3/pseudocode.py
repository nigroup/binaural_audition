if we do reject:


ldl_batch()
    all_data_taken_from_buffer:
        clear_buffer()

    if can_still_take_batch_from_indexes (enough data in buffer):
        return_batch
    else:

        if self._nothing_left():  # no new data
            if self.use_every_timestep:
                pass
            if not self.next_epoch():
                return None, None
        self.fill_buffer()
        indexes = not_rejected_positions()
        return self.ldl_batch()






not_rejected_positions():
    max_position = first_null_in_a_buffer_row (==buffer_length if buffer is completely full)
    indexes = create_permutation_indexes(0,max_position)
    indexes = remove_reject(indexes)