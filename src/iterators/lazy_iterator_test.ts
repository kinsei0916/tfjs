/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */

// tslint:disable:max-line-length
import {TensorContainerArray, TensorContainerObject} from '@tensorflow/tfjs-core/dist/types';
import {DataElement} from '../types';
import {iteratorFromIncrementing, iteratorFromZipped, LazyIterator} from './lazy_iterator';
import {iteratorFromConcatenated} from './lazy_iterator';
import {iteratorFromConcatenatedFunction} from './lazy_iterator';
import {iteratorFromFunction, iteratorFromItems} from './lazy_iterator';
// tslint:enable:max-line-length

export class TestIntegerIterator extends LazyIterator<number> {
  currentIndex = 0;
  data: number[];

  constructor(protected readonly length = 100) {
    super();
    this.data = Array.from({length}, (v, k) => k);
  }

  async next(): Promise<IteratorResult<number>> {
    if (this.currentIndex >= this.length) {
      return {value: null, done: true};
    }
    const result = this.data[this.currentIndex];
    this.currentIndex++;
    // Sleep for a millisecond every so often.
    // This purposely scrambles the order in which these promises are resolved,
    // to demonstrate that the various methods still process the stream
    // in the correct order.
    if (Math.random() < 0.1) {
      await new Promise(res => setTimeout(res, 1));
    }
    return {value: result, done: false};
  }
}

describe('LazyIterator', () => {
  it('collects all stream elements into an array', done => {
    const readIterator = new TestIntegerIterator();
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(100);
        })
        .then(done)
        .catch(done.fail);
  });

  it('reads chunks in order', done => {
    const readIterator = new TestIntegerIterator();
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(100);
          for (let i = 0; i < 100; i++) {
            expect(result[i]).toEqual(i);
          }
        })
        .then(done)
        .catch(done.fail);
  });

  it('filters elements', done => {
    const readIterator = new TestIntegerIterator().filter(x => x % 2 === 0);
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(50);
          for (let i = 0; i < 50; i++) {
            expect(result[i]).toEqual(2 * i);
          }
        })
        .then(done)
        .catch(done.fail);
  });

  it('maps elements', done => {
    const readIterator = new TestIntegerIterator().map(x => `item ${x}`);
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(100);
          for (let i = 0; i < 100; i++) {
            expect(result[i]).toEqual(`item ${i}`);
          }
        })
        .then(done)
        .catch(done.fail);
  });

  it('batches elements', done => {
    const readIterator = new TestIntegerIterator().batch(8);
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(13);
          for (let i = 0; i < 12; i++) {
            expect(result[i]).toEqual(
                Array.from({length: 8}, (v, k) => (i * 8) + k));
          }
          expect(result[12]).toEqual([96, 97, 98, 99]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be limited to a certain number of elements', done => {
    const readIterator = new TestIntegerIterator().take(8);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('is unaltered by a negative or undefined take() count.', done => {
    const baseIterator = new TestIntegerIterator();
    const readIterator = baseIterator.take(-1);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual(baseIterator.data);
        })
        .then(done)
        .catch(done.fail);
    const baseIterator2 = new TestIntegerIterator();
    const readIterator2 = baseIterator2.take(undefined);
    readIterator2.collectRemaining()
        .then(result => {
          expect(result).toEqual(baseIterator2.data);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can skip a certain number of elements', done => {
    const readIterator = new TestIntegerIterator().skip(88).take(8);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([88, 89, 90, 91, 92, 93, 94, 95]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('is unaltered by a negative or undefined skip() count.', done => {
    const baseIterator = new TestIntegerIterator();
    const readIterator = baseIterator.skip(-1);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual(baseIterator.data);
        })
        .then(done)
        .catch(done.fail);
    const baseIterator2 = new TestIntegerIterator();
    const readIterator2 = baseIterator2.skip(undefined);
    readIterator2.collectRemaining()
        .then(result => {
          expect(result).toEqual(baseIterator2.data);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created from an array', done => {
    const readIterator = iteratorFromItems([1, 2, 3, 4, 5, 6]);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([1, 2, 3, 4, 5, 6]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created from a function', done => {
    let i = -1;
    const func = () =>
        ++i < 7 ? {value: i, done: false} : {value: null, done: true};

    const readIterator = iteratorFromFunction(func);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([0, 1, 2, 3, 4, 5, 6]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created with incrementing integers', done => {
    const readIterator = iteratorFromIncrementing(0).take(7);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([0, 1, 2, 3, 4, 5, 6]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be concatenated', done => {
    const a = iteratorFromItems([1, 2, 3]);
    const b = iteratorFromItems([4, 5, 6]);
    const readIterator = a.concatenate(b);
    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual([1, 2, 3, 4, 5, 6]);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created by concatenating streams', done => {
    const a = new TestIntegerIterator();
    const b = new TestIntegerIterator();
    const readIterator = iteratorFromConcatenated(iteratorFromItems([a, b]));
    readIterator.collectRemaining()
        .then(result => {
          expect(result.length).toEqual(200);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created by concatenating streams from a function', done => {
    const readIterator = iteratorFromConcatenatedFunction(
        () => ({value: new TestIntegerIterator(), done: false}), 3);
    const expectedResult: number[] = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 100; j++) {
        expectedResult[i * 100 + j] = j;
      }
    }

    readIterator.collectRemaining()
        .then(result => {
          expect(result).toEqual(expectedResult);
        })
        .then(done)
        .catch(done.fail);
  });

  it('can be created by zipping an array of streams', async done => {
    try {
      const a = new TestIntegerIterator();
      const b = new TestIntegerIterator().map(x => x * 10);
      const c = new TestIntegerIterator().map(x => 'string ' + x);
      const readStream = iteratorFromZipped([a, b, c]);
      const result = await readStream.collectRemaining();
      expect(result.length).toEqual(100);

      // each result has the form [x, x * 10, 'string ' + x]

      for (const e of result) {
        const ee = e as TensorContainerArray;
        expect(ee[1]).toEqual(ee[0] as number * 10);
        expect(ee[2]).toEqual('string ' + ee[0]);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  it('can be created by zipping a dict of streams', async done => {
    try {
      const a = new TestIntegerIterator();
      const b = new TestIntegerIterator().map(x => x * 10);
      const c = new TestIntegerIterator().map(x => 'string ' + x);
      const readStream = iteratorFromZipped({a, b, c});
      const result = await readStream.collectRemaining();
      expect(result.length).toEqual(100);

      // each result has the form {a: x, b: x * 10, c: 'string ' + x}

      for (const e of result) {
        const ee = e as TensorContainerObject;
        expect(ee['b']).toEqual(ee['a'] as number * 10);
        expect(ee['c']).toEqual('string ' + ee['a']);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  it('can be created by zipping a nested structure of streams', async done => {
    try {
      const a = new TestIntegerIterator().map(x => ({'a': x, 'constant': 12}));
      const b = new TestIntegerIterator().map(
          x => ({'b': x * 10, 'array': [x * 100, x * 200]}));
      const c = new TestIntegerIterator().map(x => ({'c': 'string ' + x}));
      const readStream = iteratorFromZipped([a, b, c]);
      const result = await readStream.collectRemaining();
      expect(result.length).toEqual(100);

      // each result has the form
      // [
      //   {a: x, 'constant': 12}
      //   {b: x * 10, 'array': [x * 100, x * 200]},
      //   {c: 'string ' + x}
      // ]

      for (const e of result) {
        const ee = e as TensorContainerArray;
        const aa = ee[0] as TensorContainerObject;
        const bb = ee[1] as TensorContainerObject;
        const cc = ee[2] as TensorContainerObject;
        expect(aa['constant']).toEqual(12);
        expect(bb['b']).toEqual(aa['a'] as number * 10);
        expect(bb['array']).toEqual([
          aa['a'] as number * 100, aa['a'] as number * 200
        ]);
        expect(cc['c']).toEqual('string ' + aa['a']);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });

  /**
   * This test demonstrates behavior that is intrinsic to the tf.data zip() API,
   * but that may not be what users expect.  This may merit a onvenience
   * function (e.g., maybe flatZip()).
   */
  it('zipping DataElement streams requires manual merge', async done => {
    function naiveMerge(xs: DataElement[]): DataElement {
      const result = {};
      for (const x of xs) {
        // For now, we do nothing to detect name collisions here
        Object.assign(result, x);
      }
      return result;
    }

    try {
      const a = new TestIntegerIterator().map(x => ({'a': x}));
      const b = new TestIntegerIterator().map(x => ({'b': x * 10}));
      const c = new TestIntegerIterator().map(x => ({'c': 'string ' + x}));
      const zippedStream = iteratorFromZipped([a, b, c]);
      // At first, each result has the form
      // [{a: x}, {b: x * 10}, {c: 'string ' + x}]

      const readStream =
          zippedStream.map(e => naiveMerge(e as TensorContainerArray));
      // Now each result has the form {a: x, b: x * 10, c: 'string ' + x}

      const result = await readStream.collectRemaining();
      expect(result.length).toEqual(100);

      for (const e of result) {
        const ee = e as TensorContainerObject;
        expect(ee['b']).toEqual(ee['a'] as number * 10);
        expect(ee['c']).toEqual('string ' + ee['a']);
      }
      done();
    } catch (e) {
      done.fail();
    }
  });
});
