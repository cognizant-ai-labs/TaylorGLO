//
//  IncompleteBetaFunction.swift
//	LossferatuCore
//
//  Created by Santiago Gonzalez on 4/12/20.
//	Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

// Translated into Swift from: https://github.com/codeplea/incbeta/blob/master/incbeta.c

/*
 * zlib License
 *
 * Regularized Incomplete Beta Function
 *
 * Copyright (c) 2016, 2017 Lewis Van Winkle
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

import Foundation

fileprivate let STOP = 1.0e-8
fileprivate let TINY = 1.0e-30

func incbeta(_ a: Double, _ b: Double, _ x: Double) -> Double {
	
	if (x < 0.0 || x > 1.0) { return Double.nan }

    /*The continued fraction converges nicely for x < (a+1)/(a+b+2)*/
    if (x > (a+1.0)/(a+b+2.0)) {
        return (1.0-incbeta(b,a,1.0-x)) /*Use the fact that beta is symmetrical.*/
    }

    /*Find the first part before the continued fraction.*/
    let lbeta_ab = lgamma(a)+lgamma(b)-lgamma(a+b)
    let front = exp(log(x)*a+log(1.0-x)*b-lbeta_ab) / a

    /*Use Lentz's algorithm to evaluate the continued fraction.*/
    var f = 1.0
	var c = 1.0
	var d = 0.0

	var m: Double = 0.0
	for i in 0..<2000 {
		m = Double(i / 2)

		let numerator: Double = {
			if i == 0 {
				return 1.0 /*First numerator is 1.0.*/
			} else if i % 2 == 0 {
				let numnum: Double = m*(b-m)*x
				let numdenom: Double = (a+2.0*m-1.0)*(a+2.0*m)
				return numnum/numdenom /*Even term.*/
			} else {
				let numnum: Double = (a+m)*(a+b+m)*x
				let numdenom: Double = (a+2.0*m)*(a+2.0*m+1.0)
				return -numnum/numdenom /*Odd term.*/
			}
		}()
        

        /*Do an iteration of Lentz's algorithm.*/
        d = 1.0 + numerator * d
		if (fabs(d) < TINY) { d = TINY }
        d = 1.0 / d

        c = 1.0 + numerator / c
		if (fabs(c) < TINY) { c = TINY }

        let cd = c*d
        f *= cd

        /*Check for stop.*/
        if (fabs(1.0-cd) < STOP) {
            return front * (f-1.0)
        }
    }

	
	return Double.nan /*Needed more loops, did not converge.*/
}
