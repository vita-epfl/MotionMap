![MotionMap](https://github.com/vita-epfl/MotionMap/blob/main/motionmap.png)


# MotionMap: Representing Multimodality in Human Pose Forecasting

<a href="https://arxiv.org/pdf/2412.18883"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.18883-%23B31B1B?logo=arxiv&logoColor=white" style="width: auto; height: 25px;"></a>
<a href="https://vita-epfl.github.io/MotionMap"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white" style="width: auto; height: 25px;"></a>
<a href="https://arxiv.org/pdf/2412.18883"><img alt="arXiv" src="https://img.shields.io/badge/CVPR%202025-OpenReview%20(link)-black?style=flat&logo=data%3Aimage%2Fjpeg%3Bbase64%2C%2F9j%2F4AAQSkZJRgABAQAAAQABAAD%2F2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4%2BJS5ESUM8SDc9Pjv%2F2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv%2FwAARCAEiAcIDASIAAhEBAxEB%2F8QAGwABAAIDAQEAAAAAAAAAAAAAAAYHAQQFAwL%2FxABDEAABAwICAwoMBQQDAAMAAAAAAQIDBAUGETFBURIVISJhgZGTstETFBYjNTZUcXJzscEyUlWCoUJikvAkM0NEU2P%2FxAAZAQEAAwEBAAAAAAAAAAAAAAAAAwQFAgH%2FxAAnEQEAAgIBAwQDAQEBAQAAAAAAAQIDERIEMlETFCExIjNBcWEjQv%2FaAAwDAQACEQMRAD8AuYAAAAAAAAAAADymqIaaNZJ5WRsTS565IB6gjlbjKihzbSRvqXfm%2FC3pXh%2Fg4dTiy61GaMkZTtXVG3h6VJq4L2RzlrCfHk%2Bqp4%2FxzxN%2BJ6IVnNW1dQuc1VNJ8T1U18k2ITR0vmUfr%2BIWhvlQe20%2FWt7z7bW0r%2BBlTC73SIpVmSbE6BkmxOg99rHl568%2BFsIqKnAuach9FVRVE8C5wzSRqn5Hqh0abE12pv8A5PhW7JWo7%2BdJxPS2%2Fkuozx%2FYWICK0eNYnKja2mdH%2FfEu6To0kgpK%2Bkr493SzslTXuV4U96aUILY7V%2B4SxetvptAwhk4dAAAAAAAAAAAAAAAAAAAHlLPFTs3c0rI255ZvciJ%2FJ6kcxr6Kh%2BenZU6pXlaIc2nUbdjfKg9up%2Btb3md8qD26n61veVfkmxBkmxC57WPKD158LQ3yoPbqfrW943yoPbqfrW95V%2BSbEGSbEHtY8nrz4WhvlQe3U%2FWt7xvlQe3U%2FWt7yr8k2IMk2IPax5PXnwtDfKg9up%2Btb3jfKg9up%2Btb3lX5JsQZJsQe1jyevPhaG%2BVB7dT9a3vG%2BVB7dT9a3vKvyTYgyTYg9rHk9efC0N8qD26n61veN8qD26n61veVfkmxBkmxB7WPJ68%2BFob5UHt1P1re8b5UHt1P1re8q%2FJNiDJNiD2seT158LQ3yoPbqfrW95jfKg9up%2Btb3lYZJsQZJsQe1jyevPha0csczEkie17F0OauaKfZx8K%2Br8Hvd2lOwhTtHGZhYrO42yADl6AAAAAABgDJ8ue1jVc5yNaiZqqrkiGvX3Cnt1Ms9TJuWpoTW5diJrIJd79VXZ6tcqxU6LxYkXTyrtUlx4rX%2FwAR3vFXduuMI4t1DbmpK%2FQsrvwp7k1kUqqyprpfC1Uz5Xat0vAnuTUeINCmOtPpVtebfYACRwAAAAAAAAH1FLJDIkkUjo3t0OauSofIAk9rxhLEqRXFvhGf%2FaxOMnvTWSymqoauFJqeVskbtDmqVYbdvudVbJ%2FC00mWf4mLwtf70KuTp4n5r8J6ZZj4lZpk0LPdI7tR%2BHZG6NUXcua5OBF5F1m%2BUZiYnUrMTv5gAB49AAAAAAAAAAAAAAjmNfRUPz07KkjI5jX0VD89OypLh%2FZDjJ2yhIANRRAAAAAAAAAAAAAAAAAABYGFfV%2BD3u7SnYQ4%2BFfV%2BD3u7SnYQycnfK%2FTthkAHDoAAAAAYNO53KC10jp515GsTS9diHtV1UNHTPqJ3bmONM1Urq63Oa61jp5eK1OCNmfAxO%2FaTYcU3n%2FiPJfjD4uNyqLnVLPUO5GsTQxNiGqAaURERqFOZ39gAPXgAAAAAAAAAAAAAHWsViku027fmylYvHfrcv5U%2FwB4BYrFLdpt0%2FNlMxeO%2FwDNyJy%2FQn0EEVNC2GFiMjYmTWpoRCrmzcfxr9psePfzLMEMdPC2GFiMjYmTWpoRD0MGSgtgAAAAAAAAAAAAAAABHMa%2Biofnp2VJGRzGvoqH56dlSXD%2ByHGTtlCQAaiiAAAASqy4ZoLjaoaqZ0ySP3We5fknAqpsOL3ikbl1Ws2nUIqCc%2BRlr%2FNUdYncPIy1%2FmqOsTuIvc0SejZBgTnyMtf5qjrE7h5GWv8ANUdYncPc0PRsgwJz5GWv81R1idw8jLX%2Bao6xO4e5oejZBgTnyMtf5qjrE7h5GWv81R1idw9zQ9GyDAnPkZa%2FzVHWJ3DyMtf5qjrE7h7mh6NnvhX1fg97u0p2ENegoYrdSMpYN14NmeW6XNeFczZKF53aZWqxqNAAOXoAABgycbEt0W3W1UjdlPPxGcm1eb7nVazadQ8mYiNyjuKLwtdVrSQu%2FwCPA7JVRfxu1rzaDggGpSsVrqFG1ptO5AAduQAAD3pKKprpfB0sD5Xa8k4E966EO3ZMKyVaNqa7dRQrwtjTgc%2F37E%2FkmNPTQ0sKQwRNjYmhrUyQrZOoivxX5TUxTPzKKUeCpXojqyqSP%2ByJM16VOtDhK0xJxoXyrtfIv2yO0hkqTmvb%2BrEY6x%2FHMTDtoRMvEIuhTylwtaJEXKl8Gu1j1Q7AOed%2FL3jXwilXglmSrR1bkXU2VM06UI9X2iutq%2F8AJgVrNUjeFq8%2FeWWYcxr2q1zUc1UyVFTNFJa9RePv5R2xVn6VQdWxWOW7T7p2bKZi8d%2B3kTl%2BhIa3B9JPVMlp3rTxq7OSNqZoqf27DvU9PFSwNhhYjI2Jk1qaiW%2FURx%2FH7cVw%2FPyQQRU0DIYWIyNiZNamo9AZKSyAAAAAAAAAAAAAAAAAAARzGvoqH56dlSRkcxr6Kh%2BenZUlw%2Fshxk7ZQkAGoogAAFg4W9Xqb93aUr4sHC3q9Tfu7SlXqeyE%2BHudgAFBaAAAAAAAAAAAAAAAAAABgrvEVw3wu8itdnFD5tnNpXnUmt7rfELTUTouT0buWfEvAhWxc6an3ZXzW%2FgAC6rAAAEqwxh5Ho24VrM00wxrr%2FuX7HNw5aN867dStzp4cnP%2FALl1NLARMkyTQVOoy6%2FGFjFTf5SGQCisgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABHMa%2Biofnp2VJGRzGvoqH56dlSXD%2ByHGTtlCQAaiiAAAWDhb1epv3dpSviwcLer1N%2B7tKVep7IT4e52AAUFoAAAAAAAAAAAAAAAAAMARXG1VlHTUiL%2BJVkcnu4E%2BqkROziufw19kbnwRMaxOjNfqcY08NdUhSyTu0gAJkYOFdCZrs2g6mHKTxy9wNcmbI18I7m0fzkc2njEy9iNzpNLJb0tlsigy84qbqRdrl093MdEwZMmZmZ3K%2FEajQADx6AAADBpOvVsa5Wur6dFauSp4ROA9iJn6eTMQ3gaG%2Fdr%2FAFCn6xBv3a%2F1Cn6xD3jbwco8t8Ghv3a%2F1Cn6xBv3a%2F1Cn6xBxt4OUeW%2BDQS92tVyS4U%2Ba%2F8A6IbycJ5MTH2RMSyADx6AAAAAAAAAHnPPFTRLLPK2ONNLnLkiAegNDfu1%2FqFP1iDfu1%2FqFP1iHXG3h5yjy3waG%2Fdr%2FUKfrEG%2Fdr%2FUKfrEHG3g5R5b5HMa%2Biofnp2VOpv3a%2F1Cn6xDj4vmiqLLTywyNkjdOmTmrmi8CkmKJi8bcXmJrKGgA01IAAAsHC3q9Tfu7SlfE3w7dKCmsdPFPWQxyN3WbXPRFTjKVupiZp8JsM6skQNDfu1%2FqFP1iDfu1%2FqFP1iFHjbwtco8t8Ghv3a%2F1Cn6xBv3a%2F1Cn6xBxt4OUeW%2BDQ37tf6hT9YhtwTxVMSSwSNkjdoc1c0U8mJj7ImJegAPHoAAAAAAAAYMmAKzu8nhbxWP2zO%2FhcvsaZ6VDt3VTOXXI5f5U8zXrGohnz9gAOngSrBECK%2BrqF0ojY0%2Bq%2FYipNsFsRLTK%2FLhdOv8IhB1E6xylxR%2BSRgAzVwAAAAAYUgOKLb4jdFljblDUcduWhHf1J9%2BcnxzMQW3fK1SRsTOWPjx%2B9NXOnATYb8Lo8leVVdAA01IAAAsDDFy8ftTWvdnNBxH7VTUvR9CvzqYduO911Yr1yim83JyZ6F5lIc1OdP8SY7cbLEMmEMmYugAAAAAAAMEPxncN3NFb2O4Geck966E6OHnJXVVEdJTS1Eq5Mjarl5isamokq6qWol%2FHK5XLychZ6am7cvCHNbUaeQANBUAAB6U8ElTURwRJm%2BRyNb71JXimljosPUlNEnEjla1OXirwmrg23%2BFqZK96cWLiR%2FEuleZPqdHGnoqH56dlSpe%2B8ta%2BE9a6xzKEgAtoAAAAAAAAAAACf4T9X4fif2lIAT%2FAAn6vw%2FE7tKVup7E2HudoAGetgAAAAAAABgyYAqqdMqiVF1Pd9VPg2LizwVzqmZZbmZ%2F1U1zYj6Z8%2FYAD14E4wYudmemyd30Qg5L8ESotPVw58LXtfl70y%2BxX6iP%2FNLh7kqABnLgAAAAAGNRkwBX%2BJrb4hdXPY3KKo47diLrTp4ec45YeIratxtT2sbnLFx4%2BVU0pzoV4aWC%2FKn%2FAGFPJXjYABOiAABYOGrlvham7t2c0PEk5di86HXK9w1ct77qxHuyhn82%2Fk2L0%2FUsJDMzU4XXcduVWQAQpAAADBk%2BJHtjY571ya1FVVXUgEZxncNxDHQMXhk48nwpoTnX6EPNq5VrrhcJqp3%2FAKO4qbG6k6DVNTFThSIUb25W2AAlcBlrXPcjGJunOXJETWpg7uErf41c1qXpnHTJuve5dH3U4vbjWZdVjlOkvtVC23W6GlbpY3jLtculek5ONfRUPz07KkiI7jX0VD89OypnYp3kiVy8apKEgA1FEAAAAAAAAAAAn%2BE%2FV%2BH4ndpSAE%2Fwn6vw%2FE%2FtKVup7E2HudoAGetgAAAAAAABgyAK6xLD4G%2F1SZcD1R6c6HLJNjam3NXTVKJwPYrF96Lmn1IyamKd0iVG8atIACVwHcwjVJT3lInLk2dis504U%2B5wz7gmfTzxzxrk%2BNyOb70OL15VmHVZ1O1qmTwpKmOspIqmJc2StRyHuZK%2BAAAAAAAAwV7iS3b33V6sblDP5xmxNqdP1LCORiS274Wp%2B4bnND5yPl2pzoTYb8Lo8leVVfAA01IAAAsPDty3xtbHPdnNF5uTlVNC86FeHYwzcfELq1j3ZRVGTHci6l6fqQZ6cqf4lxW42WCDCGTNXAAAYI%2Fi%2B4eLW5KRjspKlcl5GJp%2ByEgXgK3vlw3yus0yLnG1dxH8Kd%2FCpPgpyvvwiy21VzwAaSmAAAWNYLfvdaoonJlK%2FjyfEurm0EPw3b%2FH7szdpnFD5x%2FLloTp%2BhYRS6m%2F%2FwArOGv9ZI5jX0VD89OypIyOY19FQ%2FPTsqQYf2QlydsoSADUUQAACT2fC9JcbXFVyzzNfJnmjcsuBVTYRgsHC3q9Tfu7SlfqLTWu4S4oiZ%2BWl5E0PtNR0t7h5E0PtNR0t7iSgp%2Btk8rPp18I15E0PtNR0t7h5E0PtNR0t7iSgetk8np18I15E0PtNR0t7jtW2gjttEylic5zGKq5u0rmuZtg5te1viZexWI%2BgAHDoAAAAAAAAAAHFxTR%2BNWWVzUzfAqSJ7k0%2FwAZkALXe1r2ua5M2uTJU2oVjcaN1vuE1K7%2FAM3cVdrdKL0F3pbfE1Vs1fnbWABcVwAASnB91RjnW2Z3A5VdCq7dbfv0kwKnY90b2vY5WuaqK1yaUUsGwXtl2pcnqjamNPON2%2F3JyFHqMWp5QtYr7%2FGXXBhDJUTgAAAAAYMgCusRW7e66vRjcopvOR8melOZTllgYmt3j9qc5jc5YPOM5U1p0FfmngvzopZK8bAAJkYAALFw%2Fcd8rXHI5c5Y%2BJJ70186cJ1CAYWuXiN0SJ7soqjJjtiO%2FpX7c5PjLzU4XXcduVWQDBEkcfE9w8RtL2sdlLP5tnJtXo%2BpX518TXDx67Pa12cVP5tvKuten6HINLBTjT%2FVPJblYABOiADestAtyukNOqcTPdSfCmnu5zyZiI3L2I3Okwwtb%2FErU2R7cpajju5E1J0fU7ZhEREyRMkQyZNrTaZmV%2BsajQRzGvoqH56dlSRkcxr6Kh%2BenZU7w%2Fshzk7ZQkAGoogAAFg4W9Xqb93aUr4sHC3q9Tfu7SlXqeyE%2BHudgAFBaAAAAAAAAAAAAAAAAAABgjGMrZ4SBlwjbxouLJl%2BXUvMv1JQfEsbJo3RyNRzHorXIutDulppbbm1eUaVSDdu9tfaq99O7NWfijd%2BZv8AvAaRqxMTG4UZjU6kAB68D0pqmakqGTwSLHIxc0ch5g8ep%2FZMRQXNqRSqkNUicLFXgdyt7jslToqouaLkqaFQ79sxbV0iJHVt8ajTWq5PTn185TydPP3RYpm%2Flk6By6PEVsrURGVLY3r%2FAES8Vf54DpNcjkzaqKm1CpNZr9p4mJ%2Bn0DAzPHrINGrvNvokXw9XGjk%2FpRd07oQ1LZiSmulc%2BmijezJu6Y5%2F9e3g1HXC2t6%2BHPKN6djIrvEVtS23R7WJlDLx4%2BTanMpYpxcT27x%2B1Oexuc0HHZyprTo%2BhJgvwu5yV5VQAAGmpAAAc%2BXKWNYbjvna45nL51vEl%2BJNfPpK5O3hS4%2BJXRIHuyiqcmryO1L9ucgz05U34S4rasnxzb5cN7bVLMi5SKm4j%2BJf9z5jokGxdcPGbilKx2cdMmS8r109CcBSxU53iFnJbjVwP5ABqKIAABNsHW%2FwFA6se3j1C8XkYmjpXhIjb6N9wroaVn%2Fo7JV2JrXoLOijZDG2NibljERrU2IhU6m%2Bo4wnw1%2BdvsAFFaCOY19FQ%2FPTsqSMjmNfRUPz07KkuH9kOMnbKEgA1FEAAAsHC3q9Tfu7SlfFg4W9Xqb93aUq9T2Qnw9zsAAoLQAAAAAAAAAAAAAAAAAABgyAOZfLQy7UKx8DZmcaJ66l2LyKV3LFJDK%2BKVisexcnNXUpaxwcR2BLjH4zTNRKpiaNHhE2e%2FYWcGXj%2BM%2FSHJj5fMIKDLmuY5WuarXNXJUVMlRTBoKgAAAAAH3FPND%2FANU0kfwPVPofAD1tpdrkiZeP1PWqeMlXUzf9tTNJ8UiqeQOeMR%2FDcmjQe1HVSUVZFUxfiicjstu1Og8QezG%2Fgj4WpTVEdTTxzxLmyRqOavIp6ZcBF8G3HdwSW%2BR3Gj48fwrpTmX6koMq9eFpherblG1c363b23WSJqZRSceP3Lq5lOaTzFVt8dtizRtzlpuOmWtv9SffmIGaGG%2FOipkrxsAAmRgRVRc0VUVNCpqAAn1NfmOw2txeqLJE3cvbtfoTp4F5yBPe6R7pHu3TnKrnLtVT6SaRIHQI9fBucjlbqVU0L%2FJ8EWPFFN%2F9SXvNtAAJUYAelPA%2BqqI4IkzfI5Gt5zx7CVYLt%2BUctwenC7zcfuTSvTwcxKjxo6aOjpYqaJOJG1Gp3nuZWS3O0yvUrxroABw6COY19FQ%2FPTsqSMjmNfRUPz07KkuH9kOMnbKEgA1FEAAAsHC3q9Tfu7SlfFg4W9Xqb93aUq9T2Qnw9zsAAoLQAAAAAAAAAAAAAAAAAAAAAGDIA4V%2Bw5HcmrUU%2BUdUiadUnIvLykHnglppnQzxujkZpa5OFC1DRudopLrFuKhnHROJI3gc3%2FdhZxZ5p8T9Ib4uXzCtQdS6YerbYqvVvhoE0SsTR701HLL1bRaNwqzExOpAAdPAAAAAAAAGxb6x9vroqpmfm3Zqm1NadBZ0MrJoWSxu3THtRzV2opVJNMHXHw1G%2Bhe7jwcLOVi9y%2FUqdTTcck%2BG2p0kioioqKmaKVve7ctsuksCJlG7jx%2FCurm0Fkajg4stvjlt8YjbnLTcbg1t1p9%2BYgwX43%2F1LlruqCgA0lMAAAAAAAAJNg23%2BFqJK97eLFxI8%2FzLpXmT6kaYxz3tYxN05yojUTWqlmWqhbbrdFSt0sbxl2uXSvSVuovxrrymxV3bbbMgGetgAAEcxr6Kh%2BenZUkZHMa%2Biofnp2VJcP7IcZO2UJABqKIAABYOFvV6m%2Fd2lK%2BLBwt6vU37u0pV6nshPh7nYABQWgAAAAAAAAAAAAAAAAAAAAAAAAwZAHyqIuk41xwtb65VkjatNKv9UacC%2B9ug7YOq2ms7h5MRP2r%2Btwtc6TNY40qWJ%2FVFp6F4TkPY%2BJ24kY5jk1OTJf5LXPGanhqG7maFkqbHtRSzXqZjuhDOGP4qwFgzYXtE2a%2BK%2BDVdcblaaj8FW934J6hn7kX7EsdTT%2Bo5w2QkEy8iKX22f%2FFp9MwVQp%2BOpqHe7cp9j33GN56VkLCJmuScK7E0k%2BhwnaIvxQvl%2BZIq%2FQ6VPQUlImVPTRRcrWIi9JxPVV%2FkOowz%2FUCosPXOtyVlMsbF%2Frl4qd5KLLhmO1zJUyVDpZ0RU4vFamfJrO8CC%2Be9o0mrirX5YTQYciORWqmaLwKi6z6MECRWt5t62y5y0%2BXm891Gu1q6OjRzGiTnFtt8atyVUbc5abhXLWzX3kGNTDfnXalkrxsAAlRgAAAADvYRt%2FjVyWpenm6ZM05Xro6OFegnSHNsFv3utUUTkyldx5PiXVzaDpmXlvzvtex141AAROwAACOY19FQ%2FPTsqSM593tUd3pWwSSujRr0eitRF1Kmv3neO0VvEy5tG6zEK2BMfIim9tm%2FxaPIim9tm%2FxaX%2FcY%2FKr6V0OBMfIim9tm%2FwAWjyIpvbZv8Wj3GPyeldDiwcLer1N%2B7tKc%2FwAiKb22b%2FFp3bbQsttDHSMe57Y8%2BM7Sua5kGfLW9dQlxUtWdy2wAVE4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADBkADAMgAYMgAAAAAA%2BXtR7Va5EVFTJUXWVrd7etsuUtNw7hF3Ua7Wro7uYssj2L7d4zQJWRt85T%2FAIstbF09GnpJ8F%2BNteUWWu6oQADSUwAADrYat%2Fj93Yrm5xQecfy7E6fockn2Frf4laWyPblLUecdyJqTo%2BpBmvxokxV5WdpDJhDJmroAAAAAGDIAAAAAABgyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB8vY17HMeiOa5MlRdaH0YArK60DrbcZaVc9y1c2Ltaug1Ca4wt3h6JtaxvHp%2BB3Kxe5fuQo1MN%2BdNqN68baAASuG%2FZKDfK6QwKmcaLu5PhTv0c5ZKJkmScBHcH2%2FwAXt7qx6ceoXi8jE0dK8PQSIzc9%2BV9eFzFXVQyAQJQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwZAHxJG2WN0b0RzXIqKi60KzudC63XCaldoYvFXa1dClnEaxjbvDUjK6NvHg4H5a2L3L9Sx09%2BNteUWWu42hhs26jdcK%2BGlb%2FAOjslXY3WvQaxLsGW%2Fcxy3B7eF%2Fm4%2Fcmlen6F3LfhWZVqV5W0lEUbYo2xsbuWMRGtTYiH2YMmUvAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADB8SxMmifFI3dMe1WuRdaKehgCtKi1zw3hba1M3rIjWLtRdC9BYtJTMo6WKmiTJkbUah8Pt9NJXx1zmZzxsVjXZ6l%2F3%2BTaJsmWbxCOlOOwAEKQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH%2F%2FZ&logoColor=white&logoSize=auto&labelColor=white&color=%23AF001E
" style="width: auto; height: 25px;"></a>
<a href="https://hub.docker.com/repository/docker/meghshukla/motionmap/"><img alt="Docker" src="https://img.shields.io/badge/Image-motionmap-%232496ED?logo=docker&logoColor=white" style="width: auto; height: 25px;"></a>
<br>

Code repository for "MotionMap: Representing Multimodality in Human Pose Forecasting". *We propose a new representation for learning multimodality in human pose forecasting which does not depend on generative models.* <br><br>

**MotionMap Saved Checkpoints: [https://drive.switch.ch/index.php/s/y9w13AnwSKy1rQe](https://drive.switch.ch/index.php/s/y9w13AnwSKy1rQe)** <br><br>



🌟 𝐌𝐨𝐭𝐢𝐨𝐧𝐌𝐚𝐩: 𝐑𝐞𝐩𝐫𝐞𝐬𝐞𝐧𝐭𝐢𝐧𝐠 𝐌𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥𝐢𝐭𝐲 𝐢𝐧 𝐇𝐮𝐦𝐚𝐧 𝐏𝐨𝐬𝐞 𝐅𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐢𝐧𝐠 is the result of our curiosity: is diffusion for X, the current trend for solving any task, the only way forward? <br>

🚶‍♂️‍➡️ Take for instance 𝐡𝐮𝐦𝐚𝐧 𝐩𝐨𝐬𝐞 𝐟𝐨𝐫𝐞𝐜𝐚𝐬𝐭𝐢𝐧𝐠, where different future motions of a person have long been simulated using generative models like Diffusion / VAEs / GANs. However, these models rely on repeatedly sampling a large number of times to generate multimodal futures. <br>
1️⃣ This is highly inefficient, since it is hard to estimate how many samples are needed to capture the likeliest modes.<br>
2️⃣ Moreover, which of the predicted futures is the likeliest future?<br>

💡 Enter MotionMap, our novel 𝐫𝐞𝐩𝐫𝐞𝐬𝐞𝐧𝐭𝐚𝐭𝐢𝐨𝐧 𝐟𝐨𝐫 𝐦𝐮𝐥𝐭𝐢𝐦𝐨𝐝𝐚𝐥𝐢𝐭𝐲. Our idea is simple, we extend heatmaps to represent a spatial distribution over the space of motions, where different maxima correspond to different forecasts for a given observation. <br>
1️⃣ MotionMap thus allows us to represent a variable number of modes per observation and provide confidence measures for different modes. <br>
2️⃣ Further, MotionMap allows us to introduce the notion of uncertainty and controllability over the forecasted pose sequence. <br>
3️⃣ Finally, MotionMap explicitly captures rare modes that are non-trivial to evaluate yet critical for safety. <br>

📈 Our results on popular human pose forecasting benchmarks show that using a heatmap and codebook can outperform diffusion, while having multiple advantages


## Table of contents
1. [Installation: Docker](#installation)
2. [Organization](#organization)
3. [Code Execution](#execution)
4. [Acknowledgement](#acknowledgement)
5. [Citation](#citation)


## Installation: Docker <a name="installation"></a>

We provide a Docker image which is pre-installed with all required packages. We recommend using this image to ensure reproducibility of our results. Using this image requires setting up Docker on Ubuntu: [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods). Once installed, we can use the provided `docker-compose.yaml` file to start our environment with the following command:  `docker compose run --rm motionmap` <br>


## Organization <a name="organization"></a>

Running `python main.py` in the `code` folder executes the code, with configurations specified in `configuration.yml`. The method has two main stages: autoencoder training and MotionMap training. This is followed by a fine-tuning stage. The outcome of running the code across all stages will be two models: `autoencoder.pt` and `motionmap.pt`. The `code` folder contains the following files:
1. `main.py`: Main file to run the code
2. `configuration.yml`: Configuration file for the code
3. `dataset.py`: Data loading and processing for the different stages
4. `config.py`: Configuration parser
5. `autoencoder.py`: Autoencoder model training, visualization and evaluation
6. `motionmap.py`: MotionMap model training and evaluation. Visualization code is included in training and evaluation.
7. `multimodal.py`: This file integrates trained MotionMap and Autoencoder models for fine-tuning, visualization and evaluation.
8. `dataloaders.py`: Data loaders for Human3.6M and AMASS.
9. `visualizer.py`: Visualization code, specifically plotting PNG and GIFs for pose sequences.
10. `utilities.py`: Utility functions for the code
11. `metrics.py`: Evaluation metrics: Diversity, ADE, FDE, MMADE, MMFDE.

In addition, we use helper functions from ```BeLFusion```, which assist in loading the dataset and contain model definitions. The ```model``` folder contains models uniquely defined for the project such as MotionMap model (based on simple heatmaps) and Uncertainty Estimation (simple MLPs).


## Code Execution <a name="execution"></a>
We first need to activate the environment. This requires us to start the container: `docker compose run --rm motionmap`, which loads our image containing all the pre-installed packages.

The main file to run the experiment is: `main.py`. Experiments can be run using `python main.py`. The configuration file `configuration.yml` contains all the parameters for the experiment.

Stopping a container once the code execution is complete can be done using:
1. `docker ps`: List running containers
2. `docker stop <container id>`
We recommend reading the documentation on Docker for more information on managing containers.

## Acknowledgement <a name="acknowledgement"></a>

We thank `https://github.com/BarqueroGerman/BeLFusion` which contains the official implementation of BeLFusion. We use their code as a starting point. We also thank Valentin Perret, Yang Gao, Yasamin Borhani and Muhammad Osama for their valuable feedback and discussions. Finally, we are grateful to the computing team, RCP, at EPFL for their support. <br>

This This research is funded by the Swiss National Science Foundation (SNSF) through the project Narratives from the Long Tail: Transforming Access to Audiovisual Archives (Grant: CRSII5 198632). The project description is available on [https://www.futurecinema.live/project/](https://www.futurecinema.live/project/).

## Citation <a name="citation"></a>

If you find this work useful, please consider starring this repository and citing this work!

```
@InProceedings{hosseininejad2025motionmap,
  title = {MotionMap: Representing Multimodality in Human Pose Forecasting},
  author = {Hosseininejad*, Reyhaneh and Shukla*, Megh and Saadatnejad, Saeed and Salzmann, Mathieu and Alahi, Alexandre},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  publisher = {IEEE/CVF}
}
```









# MotionMap


Code repository for **MotionMap: Representing Multimodality in Human Pose Forecasting**. 

---

## Status

The code is currently being prepared for uploading and will be available soon. Stay tuned for updates!

*[Please refer to the **Controllability_Demo** branch for a simple visualization on the controllability experiments!]*


---

## Learn More

For more details, you can explore:

- [**Project Page** ](https://www.epfl.ch/labs/vita/research/prediction/motionmap/)
- [**Research Paper**](https://arxiv.org/pdf/2412.18883)

---
