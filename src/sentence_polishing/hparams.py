import os
import argparse

class Hparams:
    path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

    data_path = os.path.join(path, 'data/sentence_polishing')
    pretraind_word_path = os.path.join(path, 'model/pretrained_word')
    model_path = os.path.join(path, 'model/sentence_polishing')

    print("data path : {}".format(data_path))
    print('pretrained word path : {}'.format(pretraind_word_path))
    print('model path : {}'.format(model_path))

    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32004, type=int, help='vocab size')

    # train
    parser.add_argument('--train_source', default=os.path.join(data_path, 'train.src'),
                             help="source training data")
    parser.add_argument('--train_target', default=os.path.join(data_path, 'train.tgt'),
                             help="target training data")
    parser.add_argument('--train_synonym', default=os.path.join(data_path, 'train_paraphrased_pair.txt'),
                        help="train paraphrased pair dictionary")
    # val
    parser.add_argument('--eval_source', default=os.path.join(data_path, 'dev.src'),
                             help="source evaluation data")
    parser.add_argument('--eval_target', default=os.path.join(data_path, 'dev.tgt'),
                             help="target evaluation data")
    parser.add_argument('--eval_synonym', default=os.path.join(data_path, 'dev_paraphrased_pair.txt'),
                             help="eval paraphrased pair dictionary")

    ## vocabulary
    parser.add_argument('--vocab', default=os.path.join(data_path, 'vocab.vocab'), help="vocabulary file path")
    parser.add_argument('--embedding_file', default=os.path.join(pretraind_word_path, 'sgns.sogou.word'), help="embedding file path")

    # training scheme
    parser.add_argument('--logdir', default=os.path.join(model_path, "log/tcnp"), help="log directory")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)

    parser.add_argument('--lr', default=0.00001, type=float, help="learning rate")
    parser.add_argument('--clip', default=1, help='clip_grad_norm')
    parser.add_argument('--l_alpha', default=0.9, type=float,
                        help="the weighting coefficient for trade-off between loss1 and loss2.")
    parser.add_argument('--warmup_steps', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--evaldir', default=os.path.join(model_path, "eval/tcnp"), help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=300, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=1200, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=6, type=int,
                        help="number of attention heads")
    parser.add_argument('--enc_maxlength', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--dec_maxlength', default=50, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # save model path
    parser.add_argument('--save_model', default=os.path.join(model_path, 'sentence_polishing_model.pt'), help='save model path')

    # test
    parser.add_argument('--test_source', default=os.path.join(data_path, 'test.src'),
                        help="source test data")
    parser.add_argument('--test_target', default=os.path.join(data_path, 'test.tgt'),
                        help="target test data")
    parser.add_argument('--test_synonym', default=os.path.join(data_path, 'test_paraphrased_pair.txt'),
                             help="test paraphrased pair dictionary")
    parser.add_argument('--ckpt', default=os.path.join(model_path, "log/tcnp"), help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--testdir', default=os.path.join(model_path, "test/tcnp"), help="test result dir")

    args = parser.parse_args()


