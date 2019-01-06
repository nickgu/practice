#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import tqdm

import pydev
import utils

import easy.learn_utils as lu
import easy.slot_file as sf

class MovieLensRankingFeatureExtractor:
    def __init__(self, movie_dir):
        self.movies = utils.load_movies(movie_dir)
        self.coder = lu.SlotIndexCoder()

    def begin(self, uid, iid):
        self.uid = int(uid)
        self.iid = int(iid)
        self.movie = self.movies.get(self.iid, None)

    def save(self, coder_output):
        self.coder.save(file(coder_output,'w'))

    def processes(self):
        return (self.f_uid, self.f_iid, self.f_user_genres, self.f_user_tag)

    def f_uid(self):
        return 'uid', [ self.uid ]

    def f_iid(self):
        return 'iid', [ self.iid ]

    def f_user_genres(self):
        slot = 'u_g'
        lst = []
        for genres in self.movie.genres:
            key='%s_%s' % (self.uid, genres)
            idx = self.coder.alloc(slot, key)
            lst.append( idx )
        return slot, lst

    def f_user_tag(self):
        slot = 'u_t'
        lst = []
        for tid, tag, score in self.movie.tags[:15]:
            if score < 0.3:
                continue
            key='%s_%s' % (self.uid, tag)
            idx = self.coder.alloc(slot, key)
            lst.append( idx )
        return slot, lst


class JoinDataApp(pydev.App):
    def __init__(self):
        pydev.App.__init__(self)

    def join(self):
        arg = pydev.AutoArg()
        test_num = int(arg.option('testnum', -1))
        input_filename = arg.option('f')
        movie_dir = arg.option('m')
        output_filename = arg.option('o')
        coder_output_filename = arg.option('c')
        
        data = utils.readfile(file(input_filename), test_num=test_num)

        extractor = MovieLensRankingFeatureExtractor(movie_dir)
        writer = sf.SlotFileWriter(output_filename)
        for user_id, item_id, click in tqdm.tqdm(data):
            writer.begin_instance(click)

            extractor.begin(user_id, item_id)
            ps = extractor.processes()
            for p in ps:
                slot, lst = p()
                writer.write_slot(slot, lst)

            writer.end_instance()

        extractor.save(coder_output_filename)
        writer.summary()
        


if __name__=='__main__':
    app = JoinDataApp()
    app.run()
